import csv
import os
from datetime import datetime

def save_feedback_txt(question, context, response, feedback, file_path="feedback_logs.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(file_path)

    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Timestamp", "Question", "Context", "Model Response", "Feedback"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header only if file is new
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "Timestamp": timestamp,
                "Question": question,
                "Context": context.strip(),
                "Model Response": response.strip(),
                "Feedback": feedback.strip()
            })

        print("✅ Feedback saved to feedback_logs.csv")

    except Exception as e:
        print(f"❌ Failed to save feedback: {str(e)}")
