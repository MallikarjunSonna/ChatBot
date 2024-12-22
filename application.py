from flask import Flask, render_template, request
import joblib
import logging

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('chatbot_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Set up logging
logging.basicConfig(filename='chatbot_logs.log', level=logging.INFO)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/process", methods=["POST"])
def process_input():
    try:
        # Get the user input from the form
        user_input = request.form.get("user_input")

        # Handle empty input
        if not user_input.strip():
            return render_template("response.html", user_input="No input provided.", chatbot_response="Please provide a valid query.")

        # Preprocess and vectorize the input
        query_vector = tfidf.transform([user_input]).toarray()
        predicted_intent = model.predict(query_vector)[0]

        # Add intent-based responses
        intent_responses = {
            "cancel_order": "Sure! You can cancel your order by going to your account settings.",
            "change_order": "You can change your order by contacting customer service.",
            "track_order": "You can track your order on the orders page.",
            "default": "I'm not sure how to help with that. Please contact support."
        }

        # Get chatbot response based on intent
        chatbot_response = intent_responses.get(predicted_intent, intent_responses["default"])

        # Log the input and intent
        logging.info(f"User Query: {user_input}, Predicted Intent: {predicted_intent}")

        # Render the response page
        return render_template("response.html", user_input=user_input, chatbot_response=chatbot_response)

    except Exception as e:
        # Handle errors gracefully
        logging.error(f"Error occurred: {str(e)}")
        return render_template("response.html", user_input="An error occurred.", chatbot_response="Something went wrong. Please try again.")
        
if __name__ == "__main__":
    app.run(debug=True)
