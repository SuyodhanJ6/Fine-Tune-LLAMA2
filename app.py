# app.py

import streamlit as st
from lama2FineTune.components.fine_tune import FineTuner
from lama2FineTune.constants import MODEL_ADAPTER_NAME, NUM_EPOCHS


def main():
    st.title("LLAMA2 Fine-Tuning App")

    # Get user input for model name and number of epochs
    model_name = st.text_input("Enter Model Name", value=MODEL_ADAPTER_NAME)
    num_epochs = st.number_input("Enter Number of Epochs", min_value=1, value=NUM_EPOCHS)

    # Display fine-tuning button
    if st.button("Fine-Tune Model"):
        
        # Perform fine-tuning
        st.info(f"Fine-tuning model {model_name} for {num_epochs} epochs. This may take some time...")

        # Initialize and fine-tune the model using FineTuner class
        fine_tuner = FineTuner(model_name=model_name, num_epochs=num_epochs)
        fine_tuner.fine_tune()

        st.success("Fine-tuning completed successfully!")
        #Display generated output after fine-tuning
        sample_query = "### Instruction: Who is Prashant Malge? \n\n ### Response:"
        completion = FineTuner.model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
        st.subheader("Generated Output (after fine-tuning):")
        st.text(completion)

        # except Exception as e:
        #     # Handle exceptions and display error message
        #     st.error(f"An error occurred during fine-tuning: {str(e)}")

if __name__ == "__main__":
    main()