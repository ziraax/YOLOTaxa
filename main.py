from data_preprocessing import full_preprocessing, get_classes
from config import CONFIG
import wandb
import traceback

def main():
    try:
        # Initialize a single W&B run
        wandb.init(
            project=CONFIG['project_name'],
            job_type="preprocessing with parallelization",
            config=CONFIG,
            tags=["preprocessing"]
        )
        
        # Run preprocessing
        full_preprocessing()
        
        # Log dataset artifact
        dataset_artifact = wandb.Artifact(
            name="processed_dataset",
            type="dataset",
            description=f"Processed dataset with scale bars removed and augmentations",
            metadata={
                "classes": len(get_classes()),
                "augmentation_threshold": CONFIG['augmentation_threshold'],
                "grayscale_conversion": CONFIG['convert_grayscale']
            }
        )
        dataset_artifact.add_dir(CONFIG['yolo_dataset_path'])
        wandb.log_artifact(dataset_artifact)
        
    except Exception as e:
        # Log error before finishing run
        wandb.alert(
            title="Preprocessing Failed",
            text=f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        )
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()