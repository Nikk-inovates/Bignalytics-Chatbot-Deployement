import csv
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)

# Feedback storage directory
FEEDBACK_DIR = "logs"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def save_feedback_txt(question, context, response, feedback, file_name="feedback_logs.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = os.path.join(FEEDBACK_DIR, file_name)

    file_exists = os.path.isfile(file_path)

    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Timestamp", "Question", "Context", "Model Response", "Feedback"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "Timestamp": timestamp,
                "Question": question.strip(),
                "Context": context.strip(),
                "Model Response": response.strip(),
                "Feedback": feedback.strip()
            })

        logging.info("✅ Feedback saved to %s", file_path)

    except Exception as e:
        logging.error("❌ Failed to save feedback: %s", str(e))
