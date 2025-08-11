import sys
import boto3
import time, io, os, json
import zipfile
from datetime import datetime
from awsglue.utils import getResolvedOptions

# Define the expected arguments
args = getResolvedOptions(sys.argv, ['BUCKET_NAME', 'SOURCE_PREFIX', 'PROCESSED_PREFIX', 'DYNAMO_TABLE', 'TOPIC_ARN', 'ROLE_ARN'])

BUCKET = args['BUCKET_NAME']
PREFIX = args['SOURCE_PREFIX']
PROCESSED_PREFIX = args['PROCESSED_PREFIX']
DYNAMO_TABLE = args['DYNAMO_TABLE']
TOPIC_ARN = args['TOPIC_ARN']
ROLE_ARN =  args['ROLE_ARN']

#SUPPORTED_EXTENSIONS = ('.pdf', '.jpg', '.jpeg', '.png', '.jfif')
SUPPORTED_EXTENSIONS = ('.pdf')

dynamodb = boto3.resource('dynamodb')
ddb_table = dynamodb.Table(DYNAMO_TABLE)
s3 = boto3.client('s3')
textract = boto3.client('textract')

def list_supported_files(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    page_iter = paginator.paginate(Bucket=bucket, Prefix=prefix)

    total_files = 0
    for page in page_iter:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            lower_key = key.lower()
            '''
            if lower_key.endswith(".zip"):
                extracted_keys, zip_key = extract_supported_files_from_zips(bucket, key)
                for extracted_key in extracted_keys:
                    yield extracted_key, True, zip_key  # from_zip
                yield zip_key, 'ZIP', None  # move the zip later
            elif lower_key.endswith(SUPPORTED_EXTENSIONS):
                yield key, False, None
            '''
            if lower_key.endswith(SUPPORTED_EXTENSIONS):
                yield key, False, None
                total_files += 1
            elif not lower_key.endswith("/"):
                yield key, 'UNPROCESSABLE', None

    print(f"[SUMMARY] Total supported files found: {total_files}")

# ───── Extract supported files from zips ──────────────────────────
def extract_supported_files_from_zips(bucket, zip_key, unzip_dirname="unzipped"):
    print(f"[INFO] Extracting ZIP: s3://{bucket}/{zip_key}")
    obj = s3.get_object(Bucket=bucket, Key=zip_key)
    zip_body = obj['Body'].read()

    # Extract the prefix path of the ZIP key (everything before filename)
    prefix_path = os.path.dirname(zip_key)
    zip_base = os.path.splitext(os.path.basename(zip_key))[0]  # filename without .zip

    supported_files = []
    with zipfile.ZipFile(io.BytesIO(zip_body)) as zip_ref:
        for name in zip_ref.namelist():
            lower_name = name.lower()
            if(lower_name).endswith(SUPPORTED_EXTENSIONS):
                file_bytes = zip_ref.read(name)

                # Upload path: same folder as the zip file, under an 'unzipped' subfolder
                temp_key = f"{prefix_path}/{unzip_dirname}/{zip_base}/{os.path.basename(name)}"

                print(f"[Info] Uploading extracted: {temp_key}")
                s3.put_object(
                    Bucket=bucket,
                    Key=temp_key,
                    Body=file_bytes,
                    ContentType='application/pdf' if lower_name.endswith('.pdf') else 'image/jpeg'
                )

                supported_files.append(temp_key)
    return supported_files, zip_key

def start_textract_job(s3_key):
    response = textract.start_document_text_detection(
        DocumentLocation={
            "S3Object": {
                "Bucket": BUCKET,
                "Name": s3_key
            }
        },
        NotificationChannel={
            "SNSTopicArn": TOPIC_ARN,
            "RoleArn": ROLE_ARN
        }
    )
    return response["JobId"]

def move_s3_object(source_key, destination_prefix):
    new_key = source_key.replace(PREFIX, destination_prefix, 1)
    s3.copy_object(
        Bucket=BUCKET,
        CopySource={'Bucket': BUCKET, 'Key': source_key},
        Key=new_key
    )
    s3.delete_object(Bucket=BUCKET, Key=source_key)
    print(f"[INFO] Moved {source_key} → {new_key}")

def record_job_metadata(job_id, s3_key, from_zip, zip_key):
    ddb_table.put_item(Item={
        'job_id': job_id,
        's3_key': s3_key,
        'status': 'IN_PROGRESS',
        'timestamp': datetime.utcnow().isoformat(),
        'from_zip': from_zip,
        'zip_key': zip_key if zip_key else None,
        'file_type': os.path.splitext(s3_key)[-1].lstrip('.').lower()
    })

def main():
    files_processed = 0
    for s3_key, source_type, zip_key in list_supported_files(BUCKET, PREFIX):
        print(s3_key)
        try:
            if source_type == "ZIP" or source_type == "UNPROCESSABLE":
                move_s3_object(s3_key, PROCESSED_PREFIX)
                continue

            files_processed += 1

            if files_processed != 0 and files_processed % 15 == 0:
                print(f"[INFO] Taking a break at {files_processed} files")
                time.sleep(1)

            #if files_processed >= 2:
            #    print(f"[Info] Total Files processed: {files_processed}")
            #    break;


            job_id = start_textract_job(s3_key)
            print(f"[INFO] Started Textract job: {job_id} for {s3_key}")
            record_job_metadata(job_id, s3_key, from_zip=source_type, zip_key=zip_key)

        except Exception as e:
            print(f"[ERROR] Failed to start job for {s3_key}: {e}")

if __name__ == "__main__":
    main()
