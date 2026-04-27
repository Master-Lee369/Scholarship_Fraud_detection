from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.fraud_detection.training import (
    DEFAULT_EXTERNAL_DATASET_PATH,
    DEFAULT_EXTERNAL_DATASET_URL,
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    train_and_save_model,
)


class Command(BaseCommand):
    help = "Download scholarship reference data and train the multiclass fraud classifier."

    def add_arguments(self, parser):
        parser.add_argument(
            "--samples",
            type=int,
            default=8000,
            help="Number of synthetic scholarship applications to generate for training.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed used for dataset generation and model training.",
        )
        parser.add_argument(
            "--skip-download",
            action="store_true",
            help="Require the external scholarship CSV to already exist locally.",
        )
        parser.add_argument(
            "--external-data-path",
            default=str(DEFAULT_EXTERNAL_DATASET_PATH),
            help="Path to the external scholarship CSV file.",
        )
        parser.add_argument(
            "--external-data-url",
            default=DEFAULT_EXTERNAL_DATASET_URL,
            help="Download URL used when the scholarship CSV is missing.",
        )

    def handle(self, *args, **options):
        try:
            artifact, metrics, training_frame_path = train_and_save_model(
                sample_count=options["samples"],
                seed=options["seed"],
                external_dataset_path=Path(options["external_data_path"]),
                external_dataset_url=options["external_data_url"],
                download_if_missing=not options["skip_download"],
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS("Fraud classifier training complete."))
        self.stdout.write(f"Training rows: {artifact['training_rows']}")
        self.stdout.write(f"Accuracy: {metrics['accuracy']:.4f}")
        self.stdout.write(f"Macro F1: {metrics['macro_f1']:.4f}")
        self.stdout.write(f"Training frame: {training_frame_path}")
        self.stdout.write(f"Model artifact: {MODEL_ARTIFACT_PATH}")
        self.stdout.write(f"Metrics JSON: {METRICS_PATH}")
