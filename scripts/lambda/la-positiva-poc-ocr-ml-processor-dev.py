import json
import boto3
import botocore
import logging
import os
from datetime import datetime
import uuid
import time
from urllib.parse import urlparse

BEDROCK_MODEL_ID = "amazon.nova-micro-v1:0"          # On-demand Nova Micro
#amazon.nova-lite-v1:0
BEDROCK_INVOCATION_PARAMS = {
    "maxTokens": 4096,   # enough for long answers; change as needed
    "temperature": 0.1,  # deterministic extraction
    #"topP": 0.8
}

PROMPT_TEMPLATE = """\
Eres un asistente experto en procesar denuncias policiales en español.
Extrae y devuelve SOLO el siguiente JSON:
{{
  "contenido_denuncia": "<toda la narrativa completa de los hechos>"
}}

Reglas:
- Empieza después de encabezados como "Contenido", "Descripción de los hechos", "Acta de" o "Resumen".
- Incluye todos los párrafos, aunque estén en distintas páginas.
- Detente antes de los encabezados "Instructor", "Fdo el Instructor", "Interviniente" o "Autentificador".
- Devuelve ÚNICAMENTE JSON válido.

Documento:
<<<
{document_text}
>>>
"""


# Logging conf
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS Clients
bda_client = boto3.client('bedrock-data-automation-runtime', region_name='us-east-1')
bedrock  = boto3.client("bedrock-runtime", region_name='us-east-1')
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Env variables
REGION = os.environ['AWS_REGION']
SOURCE_BUCKET = os.environ['SOURCE_BUCKET']
RESULTS_BUCKET = os.environ['RESULTS_BUCKET']
DOCUMENTS_TABLE = os.environ['DOCUMENTS_TABLE']
BDA_PROJECT_ARN = os.environ['BDA_PROJECT_ARN']
SOURCE_PREFIX = os.environ['SOURCE_PREFIX']

# Verify BDA state
MAX_POLLS       = 60            # ~10 min at 10‑sec intervals
POLL_INTERVAL   = 10            # seconds

def lambda_handler(event, context):

    print("Init function")

    try:
        logger.info(f"Event in runtime 3.13 : {json.dumps(event)}")

        #return process_document('la-positiva-poc-ocr-ml-sync-files', 'filtered/Salesforce_062024/7019846/DP P´LACA 3202-FT_textract_id_ec533f0675a9fbb9e102504d53c38529559ca7342c93e322825739bbc935c122.pdf')

        # Process a S3 event
        if 'source' in event and event['source'] == 'aws.s3':
            print("A File from Event bridge found!")
            return handle_s3_event(event)
        else:
            logger.warning("Event type not recognized")
            return {'statusCode': 400, 'body': 'Event not supported'}

    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        raise

def handle_s3_event(event):
    """
    Process S3 event when a pdf file is uploaded
    """
    try:
        # Extract event information
        detail = event['detail']
        bucket_name = detail['bucket']['name']
        object_key = detail['object']['key']

        logger.info(f"Processing: s3://{bucket_name}/{object_key}")

        # Validate PDF prefix and sufix
        if not object_key.startswith(SOURCE_PREFIX) or not object_key.lower().endswith('.pdf'):
            logger.info("File not valid to store")
            return {'statusCode': 200, 'body': 'File Ignored'}

        # Process document
        print("\n Before process the document")
        return process_document(bucket_name, object_key)

    except Exception as e:
        logger.error(f"Error procesando evento S3: {str(e)}")
        raise

def process_document(bucket_name, object_key):
    """
    Send document to Bedrock Data Automation
    """
    try:
        # Generate unique document id
        document_id = f"{object_key.rsplit("/", 1)[-1].replace('.pdf', '')}_{uuid.uuid4().hex[:8]}"
        case_id = ""
        try:
            case_id = extract_case_id_from_key(object_key)
        except Exception as e:
            case_id = "0000000"
        case_id = extract_case_id_from_key(object_key)
        print(f"Document id: {document_id} and case id: {case_id}")


        # URIs
        input_uri = f"s3://{bucket_name}/{object_key}"
        output_uri = f"s3://{RESULTS_BUCKET}/processed/{document_id}"

        print(f"\n input_uri: {input_uri}")
        print(f"\n output_uri: {output_uri}")
        # Register in Dynamo - Initial state
        register_document(document_id, case_id, object_key, input_uri, output_uri, 'INITIATED')

        # Get Account ID
        account_id = boto3.client('sts').get_caller_identity()['Account']
        profile_arn = f'arn:aws:bedrock:{REGION}:{account_id}:data-automation-profile/us.data-automation-v1'

        #print(f"\n profile_arn bedrock: {profile_arn}")
        # Invoke BDA
        logger.info(f"Send to BDA: {input_uri}")

        print(f"\n BDA_PROJECT_ARN bedrock: {BDA_PROJECT_ARN}")
        response = bda_client.invoke_data_automation_async(
            inputConfiguration={'s3Uri': input_uri},
            outputConfiguration={'s3Uri': output_uri},
            dataAutomationConfiguration={
                'dataAutomationProjectArn': BDA_PROJECT_ARN,
                'stage': 'LIVE'
            },
            dataAutomationProfileArn=profile_arn
        )

        invocation_arn = response['invocationArn']
        logger.info(f"BDA iniciado: {invocation_arn}")

        invocation_arn = response['invocationArn']
        logger.info(f"BDA iniciado: {invocation_arn}")

        # Update Dynamo status

        register_document(document_id, case_id, object_key, input_uri, output_uri, 'PROCESSING')

        # Poll Bedrock until it finishes
        status_resp = wait_for_bda(invocation_arn)

        if status_resp["status"] == "Success":

            print("Status del BDA!!")
            print(status_resp)
            final_state = "SUCCESS"

            # Download the metadata JSON and copy results
            ocr_results, from_custom_blueprint, src_uri = fetch_results(status_resp["outputConfiguration"]["s3Uri"], document_id)
            print(f"This is the place of the document result: {src_uri}")
            #register_document(document_id, object_key, input_uri, output_uri, final_state, ocr_results, from_custom_blueprint)
        else:
            logger.error(f"BDA finished with error: "
                        f'{status_resp.get("errorType")} – '
                        f'{status_resp.get("errorMessage")}')
            final_state = "FAILED"
            register_document(document_id, case_id, object_key, input_uri, output_uri, final_state)
            raise

        print(f"\n invocation arn: {invocation_arn}")

        ### Content with LLM section ###
        try:
            ##### LLM ###
            txt_object_key = object_key.replace("filtered/", "filtered_all_words/")
            txt_object_key = txt_object_key.replace(".pdf", "_all_words.txt")

            print(f"\n txt_object_key: {txt_object_key}")

            # Read OCR text from S3
            obj = s3_client.get_object(Bucket=bucket_name, Key=txt_object_key)
            document_text = obj["Body"].read().decode("utf-8", errors="replace")
            #print(document_text)

            parsed_result_dba_url = urlparse(src_uri)
            key_result_bda_url = parsed_result_dba_url.path.lstrip("/")
            dest_key = key_result_bda_url.rsplit("/", 1)[0] + "/contenido_denuncia.json"

            print(f"Result bda url: {key_result_bda_url} and dest key {dest_key}")

            # Prepare prompt
            prompt = PROMPT_TEMPLATE.format(document_text=document_text)
            messages = [{"role": "user", "content": [{"text": prompt}]}]

            #Call Bedrock Amazon Nova Micro
            resp = bedrock.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=messages,
                inferenceConfig=BEDROCK_INVOCATION_PARAMS,
            )

            model_text = resp["output"]["message"]["content"][0]["text"]
            #print(f"Response model: {model_text}")
            payload = _coerce_to_json(model_text)

            #Build destination key (mirror path, change .txt -> .json)
            out_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

            # Write JSON to destination bucket
            s3_client.put_object(
                Bucket=RESULTS_BUCKET,
                Key=dest_key,
                Body=out_bytes,
                ContentType="application/json; charset=utf-8",
                CacheControl="no-cache",
            )

            final_result = json.loads(ocr_results)
            #print('Before updating final result')
            #print(payload['contenido_denuncia'])
            final_result['inference_result']["contenido_denuncia_from_txt"] = payload['contenido_denuncia']
            register_document(document_id, case_id, object_key, input_uri, output_uri, 'SUCCESS', json.dumps(final_result), from_custom_blueprint)

        except Exception as e:
            logger.error(f"Error procesando documento en la extracion de contenido_denuncia con LLM: {str(e)}")
            # Registrar fallo en DynamoDB
            try:
                register_document(document_id, case_id, object_key, input_uri, output_uri, 'FAILED', ocr_results, from_custom_blueprint)
            except:
                pass
            raise

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    'document_id': document_id,
                    "message": "Saved JSON to S3",
                    "dest_bucket": RESULTS_BUCKET,
                    "dest_key": dest_key,
                    'bda_invocation_arn': invocation_arn,
                    'status': final_state
                },
                ensure_ascii=False,
            ),
        }


    except Exception as e:
        logger.error(f"Error procesando documento: {str(e)}")
        try:
            register_document(document_id, case_id, object_key, input_uri, dest_key, 'FAILED')
        except:
            pass
        raise


def register_document(document_id, case_id, original_key, input_uri, output_uri, status, results=None, from_custom_blueprint=False):
    """
    Register documento in DynamoDB
    """
    table = dynamodb.Table(DOCUMENTS_TABLE)

    timestamp = datetime.now().isoformat()

    item = {
        'document_id': document_id,
        'case_id': case_id,
        'processing_timestamp': timestamp,
        'original_key': original_key,
        'input_uri': input_uri,
        'output_uri': output_uri,
        'status': status,
        'created_at': timestamp,
        'updated_at': timestamp,
        'results': results if results else ''
    }

    # Extract and get field if needed
    if results:
        try:
            result_data = json.loads(results) if isinstance(results, str) else results
            inference = result_data.get("inference_result", {})

            # Expand result keys, each one as column
            if isinstance(inference, dict) and from_custom_blueprint == True:
                item.update(inference)

        except Exception as e:
            logger.warning(f"Error parsing results for DynamoDB: {e}")

    print(f"The item to register: ")
    table.put_item(Item=item)
    logger.info(f"Document registered: {document_id}")


def extract_case_id_from_key(key):
    """
    Expected pattern:
      filtered/<any_folder>/<CASE_ID>/<filename>.pdf
    Returns CASE_ID as a string or raises ValueError if not found.
    """
    parts = key.strip("/").split("/")
    if len(parts) >= 3 and parts[2].isdigit():
        return parts[2]
    raise ValueError(f"CASE_ID not found at segment 3 in key: {key}")

def _coerce_to_json(text):
    """
    Try to parse model output as JSON.
    If the model returned plain text, wrap it under the expected key.
    """
    text = text.strip()
    try:
        data = json.loads(text)
        # Ensure expected key exists; if not, wrap it
        if isinstance(data, dict) and "contenido_denuncia" in data:
            return data
        return {"contenido_denuncia": text}
    except Exception:
        # Fallback: wrap raw text
        return {"contenido_denuncia": text}

def wait_for_bda(invocation_arn):
    """
    Poll GetDataAutomationStatus until Success | ClientError | ServiceError
    or we hit MAX_POLLS.
    """

    print("Starting TRACK BDA")

    for _ in range(MAX_POLLS):
        resp = bda_client.get_data_automation_status(
            invocationArn=invocation_arn
        )
        state = resp["status"]
        logger.info(f"Estado BDA = {state}")
        if state in ("Success", "ClientError", "ServiceError"):
            return resp
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"BDA dont finish in {MAX_POLLS * POLL_INTERVAL}s"
    )

def fetch_results(metadata_s3_uri, document_id):
    """
    Download the Bedrock metadata JSON, extract every custom_output_path,
    copy those result files into RESULTS_BUCKET, and return the new URIs.
    """
    # 1. download the metadata file
    parsed = urlparse(metadata_s3_uri)
    meta_obj = s3_client.get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/")
    )
    metadata = json.loads(meta_obj["Body"].read())
    logger.info(f"Result metadata: {json.dumps(metadata)[:400]}")

    # 2. collect every custom_output_path
    standard_paths = []
    custom_paths = []
    for asset in metadata.get("output_metadata", []):
        for segment in asset.get("segment_metadata", []):
            path = segment.get("custom_output_path")
            if path:
                custom_paths.append(path)
            else:
                standard_paths.append(segment.get("standard_output_path"))

    if not custom_paths:
        logger.warning("custom_output_path not found in metadata")

        for src_uri in standard_paths:
            parsed = urlparse(src_uri)
            meta_obj = s3_client.get_object(
                Bucket=parsed.netloc,
                Key=parsed.path.lstrip("/")
            )

            ocr_results = json.loads(meta_obj["Body"].read())
            logger.info(f"BDA results: {json.dumps(ocr_results)[:400]}")

            return json.dumps(ocr_results), False, src_uri

    print("Custom paths found!!")
    print(custom_paths)

    for src_uri in custom_paths:
        # 3. copy every result file into RESULTS_BUCKET
        parsed = urlparse(src_uri)
        meta_obj = s3_client.get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/")
        )

        ocr_results = json.loads(meta_obj["Body"].read())
        logger.info(f"BDA results: {json.dumps(ocr_results)[:400]}")


        return json.dumps(ocr_results), True, src_uri
