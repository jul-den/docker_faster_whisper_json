docker run -d --name webinar-processor-container -v \PATH\TO\source:/source -v \PATH\TO\output:/app/output -e VIDEO_DIR=/source -e OUTPUT_DIR=/app/output --network="host" webinar-processor
