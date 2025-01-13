def run_step(task_function, *args, **kwargs):
    """
    Template function for executing a step in the pipeline.
    - Loads assets
    - Calls the provided task function with the necessary arguments
    - Cleans up the environment
    """
    # Load asset (if applicable)
    print("Loading asset...")
    # You can define logic here to load any assets specific to each step

    # Execute the task
    print("Running task...")
    result = task_function(*args, **kwargs)
    
    # Clean up processing environment (e.g., clearing variables, freeing memory)
    print("Cleaning up...")
    # Include any necessary cleanup steps (e.g., `ee.data.clear()`)

    return result


def feature_extraction_step(aoi, date_range):
    def feature_extraction_task(aoi, date_range):
        # Feature extraction logic here
        # For example: extract features, process images, etc.
        return features  # Example return value

    # Use the template function to run the feature extraction step
    return run_step(feature_extraction_task, aoi, date_range)

def model_training_step(features, class_property, input_properties):
    def model_training_task(features, class_property, input_properties):
        # Model training logic here
        model = randomforest(features, class_property, input_properties)
        return model

    return run_step(model_training_task, features, class_property, input_properties)

def accuracy_assessment_step(model, validation_features, actual_property, predicted_property):
    def accuracy_assessment_task(model, validation_features, actual_property, predicted_property):
        # Accuracy assessment logic here
        metrics = compute_accuracy_metrics(model, validation_features, actual_property, predicted_property)
        return metrics

    return run_step(accuracy_assessment_task, model, validation_features, actual_property, predicted_property)
