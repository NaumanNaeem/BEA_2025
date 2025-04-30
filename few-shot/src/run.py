import os
import json
from datetime import datetime
from prompt import chain

from utils.logging import get_logger

logger = get_logger()


def run():
    # File paths.
    raw_file = "../data/raw/mrbench_v3_testset.json"
    output_file = "../data/processed/mrbench_v3_testset_processed.json"

    # Read raw data.
    with open(raw_file, "r", encoding="utf-8") as json_file:
        rows = json.load(json_file)

    enriched_data = []
    processed_ids = set()

    # If output file exists, load already processed rows.
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            enriched_data = json.load(f)
            for row in enriched_data:
                # Assuming each row has a unique 'id'
                processed_ids.add(row.get("conversation_id"))

    # Process rows that haven't been enriched.
    for idx, row in enumerate(rows, 1):
        if row.get("conversation_id") in processed_ids:
            continue

        conversation = row.get("conversation_history", "")

        enriched_models = {}

        for model, response_item in row.get("tutor_responses", {}).items():
            response = response_item.get("response", "")
            tstamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            logger.info(
                f"[{tstamp}]: Processing response {idx}-{model}/{len(rows)}: {response}"
            )

            for retry in range(5):
                try:
                    feedback = chain.invoke(
                        {"conversation": conversation, "response": response}
                    )
                    print(feedback)
                    logger.info(f"[{tstamp}]: Feedback for {model}: {feedback}")
                    annotation = dict(feedback)
                    response_item.update({"annotation": annotation})
                    enriched_models[model] = response_item
                    break
                except Exception as e:
                    logger.warning(
                        f"[Retry {retry + 1}]: Error processing question {row.get('id', 'Unknown ID')}: {str(e)}"
                    )
                    if retry < 4:
                        logger.info("Retrying...")

        row["tutor_responses"] = enriched_models
        enriched_data.append(row)

        # Save after each successful classification.
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run()
