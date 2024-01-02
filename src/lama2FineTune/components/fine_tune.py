import os
import sys
import logging
from datetime import datetime
from gradientai import Gradient

from lama2FineTune.constants import MODEL_ADAPTER_NAME, NUM_EPOCHS
from lama2FineTune.config import SAMPLES
from lama2FineTune.constants.env_varaible import GRADIENT_WORKSPACE_ID, GRADIENT_ACCESS_TOKEN
from lama2FineTune.logger import logging
from lama2FineTune.exception import Llama2Exception


class FineTuner:
    def __init__(self, model_name, num_epochs):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.gradient = None
        self.model_adapter = None

    def initialize_gradient(self):
        # Initialize Gradient AI Cloud with credentials
        self.gradient = Gradient(workspace_id=GRADIENT_WORKSPACE_ID, access_token=GRADIENT_ACCESS_TOKEN)

    def create_model_adapter(self):
        # Create model adapter with the specified name
        base_model = self.gradient.get_base_model(base_model_slug="nous-hermes2")
        model_adapter = base_model.create_model_adapter(name=self.model_name)
        return model_adapter

    def fine_tune_model(self, samples):
        # Fine-tune the model using the provided samples and number of epochs
        # for epoch in range(self.num_epochs):
        #     for sample in samples:
        #         query = sample["inputs"]
        #         response = sample["response"]
        #         self.model_adapter.fine_tune(inputs=query, targets=response)
        count = 0
        while count < NUM_EPOCHS:
            logging.info(f"Fine-tuning the model with iteration {count + 1}")
            self.model_adapter.fine_tune(samples=samples)
            count = count + 1

    def fine_tune(self):
        try:
            # Initialize logging
          
            # Initialize Gradient AI Cloud
            self.initialize_gradient()

            # Create model adapter
            self.model_adapter = self.create_model_adapter()
            logging.info(f"Created model adapter with id {self.model_adapter.id}")

            # Fine-tune the model
            self.fine_tune_model(SAMPLES)

        except Exception as e:
            # Handle exceptions using custom exception class and logging
           raise Llama2Exception(e, sys)

        # finally:
        #     # Clean up resources if needed
        #     if self.model_adapter:
        #         self.model_adapter.delete()
        #     if self.gradient:
                # self.gradient.close()

# if __name__ == "__main__":
#     # Example usage
#     fine_tuner = FineTuner(model_name=MODEL_ADAPTER_NAME, num_epochs=NUM_EPOCHS)
#     fine_tuner.fine_tune()
