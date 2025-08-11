import boto3
import time, io, os, json
from collections import defaultdict
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
from boto3.dynamodb.conditions import Key
from decimal import Decimal

BUCKET = os.environ['BUCKET_NAME']
SOURCE_PREFIX = os.environ['SOURCE_PREFIX']
TARGET_PREFIX = os.environ['TARGET_PREFIX']
TARGET_ALL_WORDS_PREFIX = os.environ['TARGET_ALL_WORDS_PREFIX']
PROCESSED_PREFIX = os.environ['PROCESSED_PREFIX']
DYNAMO_TABLE = os.environ['DYNAMO_TABLE']

KEYWORDS = {
    "telefono", "licipante", "fallecido", "denunciante", "raviado", "tipificacion", "lugar del hecho", "participante",
    "ocupante", "contenido", "detenido", "ampliacion", "interviniente", "autentificador", "comisaria pnp",
    "policia nacional", "instructor", "vehiculo(s)", "pnp", "regpol", "formalidad escrita",
    "acta de intervencion", "impresion digital", "tentificador", "viniente", "implicado", "citado", "deponente"
}
MIN_HITS = 3

dynamodb = boto3.resource('dynamodb')
ddb_table = dynamodb.Table(DYNAMO_TABLE)
s3 = boto3.client('s3')
textract = boto3.client('textract')


def get_pending_jobs():
    response = ddb_table.query(
        IndexName="status-timestamp-index",
        KeyConditionExpression=Key("status").eq("IN_PROGRESS"),
        ScanIndexForward=True  # Sort ascending (oldest first)
    )

    return response.get("Items", [])

def check_textract_results(job_id):
    return textract.get_document_text_detection(JobId=job_id)

def extract_pages_with_keywords(job_result):
    pages = defaultdict(list)
    confidences = defaultdict(list)
    for block in job_result.get("Blocks", []):
        if block["BlockType"] == "LINE":
            page = block["Page"]
            pages[page].append(block["Text"])
            confidence = block.get("Confidence", 0.0)
            confidences[page].append(Decimal(str(confidence)))

    matched = []
    matched_conf = {}

    for page_num, lines in pages.items():
        combined = " ".join(lines).lower()
        hits = sum(1 for kw in KEYWORDS if kw in combined)
        if hits >= MIN_HITS:
            matched.append((page_num, hits, lines))
            # Only now calculate average confidence
            conf_list = confidences[page_num]
            if conf_list:
                avg_conf = round(sum(conf_list) / len(conf_list), 2)
                matched_conf[str(page_num)] = avg_conf

    return matched, matched_conf

def save_filtered_output(s3_key, matches, job_id):
    ext = s3_key.rsplit(".", 1)[-1].lower()
    original_key = s3_key  # file is still in source/

    obj = s3.get_object(Bucket=BUCKET, Key=original_key)
    body = obj["Body"].read()

    if ext == "pdf":
        print("Before reading file")
        reader = PdfReader(io.BytesIO(body))
        writer = PdfWriter()
        for pg, *_ in sorted(matches):
            writer.add_page(reader.pages[pg - 1])

        out_buf = io.BytesIO()
        writer.write(out_buf)
        out_buf.seek(0)
        content = out_buf.getvalue()
        content_type = "application/pdf"
        #out_key = f"{TARGET_PREFIX}{os.path.basename(original_key).rsplit('.', 1)[0]}_keywords.pdf"
        out_key = f"{TARGET_PREFIX}{s3_key[len(SOURCE_PREFIX):].rsplit('.', 1)[0]}_textract_id_{job_id}.pdf"

    else:
        content = body
        content_type = f"image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
        #out_key = f"{TARGET_PREFIX}{os.path.basename(original_key).rsplit('.', 1)[0]}_keywords.{ext}"
        out_key = f"{TARGET_PREFIX}{s3_key[len(SOURCE_PREFIX):].rsplit('.', 1)[0]}_keywords.{ext}"

    print(f"[INFO] Uploading filtered: s3://{BUCKET}/{out_key}")
    s3.put_object(
        Bucket=BUCKET,
        Key=out_key,
        Body=content,
        ContentType=content_type,
        Metadata={
            #"source": original_key,
            "keywords": ",".join(sorted(KEYWORDS))
        },
    )
    print(f"[INFO] Uploaded filtered: s3://{BUCKET}/{out_key}")

    # === NEW SECTION: Upload .txt with matched words ===
    lines_txt = []
    for _, _, lines in sorted(matches):
        lines_txt.extend(lines)

    txt_content = "\n".join(lines_txt)
    txt_buf = io.BytesIO(txt_content.encode("utf-8"))
    txt_key = f"{TARGET_ALL_WORDS_PREFIX}{s3_key[len(SOURCE_PREFIX):].rsplit('.', 1)[0]}_textract_id_{job_id}_all_words.txt"

    s3.put_object(
        Bucket=BUCKET,
        Key=txt_key,
        Body=txt_buf,
        ContentType="text/plain",
        Metadata={
            #"source": original_key,
            "keywords": ",".join(sorted(KEYWORDS)),
        },
    )
    print(f"[INFO] Uploaded matched text: s3://{BUCKET}/{txt_key}")

def update_job_status(job_id, new_status, extra_attrs=None):
    update_expr = "SET #st = :new, updated = :now"
    expr_attr_vals = {
        ":new": new_status,
        ":now": datetime.utcnow().isoformat()
    }
    expr_attr_names = {"#st": "status"}

    if extra_attrs:
        for i, (k, v) in enumerate(extra_attrs.items()):
            placeholder = f":val{i}"
            update_expr += f", {k} = {placeholder}"
            expr_attr_vals[placeholder] = v

    ddb_table.update_item(
        Key={"job_id": job_id},
        UpdateExpression=update_expr,
        ExpressionAttributeNames=expr_attr_names,
        ExpressionAttributeValues=expr_attr_vals
    )

def move_s3_object(source_key):
    #new_key = source_key.replace(SOURCE_PREFIX, PROCESSED_PREFIX, 1)
    suffix_path = source_key[len(SOURCE_PREFIX):]
    new_key = f"{PROCESSED_PREFIX}{suffix_path}"
    s3.copy_object(Bucket=BUCKET, CopySource={"Bucket": BUCKET, "Key": source_key}, Key=new_key)
    s3.delete_object(Bucket=BUCKET, Key=source_key)
    print(f"[INFO] Moved to processed: {new_key}")

def is_temp_extracted_from_zip(s3_key):
    return s3_key.startswith(f"{SOURCE_PREFIX}unzipped/")

def lambda_handler(event, context):
    print(f"Event in runtime 3.13 : {json.dumps(event)}")

    from_zip = False

    for record in event.get("Records", []):
        sns = record.get("Sns", {})
        message_str = sns.get("Message", "{}")

        try:
            # Parse the JSON string into a Python dict
            message = json.loads(message_str)

            # Access the fields from the parsed message
            job_id = message.get("JobId")
            status = message.get("Status")
            s3_bucket = message.get("DocumentLocation", {}).get("S3Bucket")
            s3_key = message.get("DocumentLocation", {}).get("S3ObjectName")

            print(f"Job ID: {job_id}")
            print(f"Status: {status}")
            print(f"S3 Bucket: {s3_bucket}")
            print(f"S3 Key: {s3_key}")

            try:
                result = check_textract_results(job_id)
                if status == "SUCCEEDED":
                    pages = []
                    next_token = None
                    while True:
                        resp = (
                            textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
                            if next_token else result
                        )
                        pages.extend(resp["Blocks"])
                        next_token = resp.get("NextToken")
                        if not next_token:
                            break
                    print('Before extracting page with keywords')
                    matched, page_conf = extract_pages_with_keywords({"Blocks": pages})
                    if matched:
                        save_filtered_output(s3_key, matched, job_id)

                    # Clean up or move
                    if from_zip:
                        s3.delete_object(Bucket=BUCKET, Key=s3_key)
                        print(f"[INFO] Deleted temp extracted file: {s3_key}")
                    else:
                        print('Before moving to proccesed')
                        move_s3_object(s3_key)

                    update_job_status(
                        job_id,
                        "PROCESSED",
                        extra_attrs={"page_confidence": page_conf}
                    )

                elif status == "FAILED":
                    update_job_status(job_id, "FAILED")

            except Exception as e:
                print(f"[ERROR] Failed processing {job_id}: {e}")
                update_job_status(
                    job_id,
                    "FAILED",
                    extra_attrs={"failed_reason": str(e)}
                )

        except json.JSONDecodeError as e:
            print(f"Error decoding message JSON: {e}")

    return {
        'statusCode': 200,
        'body': json.dumps('Textract results processed successfully.')
    }