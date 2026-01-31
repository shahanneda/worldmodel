Upload data;
set -a; source .env; set +a; ./scripts/s3_upload_data.py

Download data;
set -a; source .env; set +a; ./scripts/s3_download_data.py