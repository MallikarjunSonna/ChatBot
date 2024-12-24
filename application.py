from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Map intents to conversational responses
responses = {
    "track_order": "You can track your order in the 'My Orders' section of your account.",
    "cancel_order": "To cancel your order, go to 'My Orders' and select 'Cancel' for the specific order.",
    "change_order": "To change your order, please contact customer support.",
    "check_refund_policy": "Our refund policy can be found in the 'Help Center' under 'Refunds'.",
    "get_invoice": "You can download your invoice from the 'Order Details' section in your account.",
    "place_order": "To place an order, add items to your cart and proceed to checkout.",
    "recover_password": "To recover your password, click on 'Forgot Password' on the login page.",
    "contact_customer_service": "You can contact customer service via the 'Help Center' or call our hotline.",
    "payment_issue": "For payment issues, please check your payment details or contact support."
}

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.json['query']
    query_vector = vectorizer.transform([user_query]).toarray()
    predicted_intent = model.predict(query_vector)[0]
    bot_response = responses.get(predicted_intent, "I'm sorry, I didn't understand that. Could you please rephrase?")
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
