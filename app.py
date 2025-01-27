from flask import Flask, request, jsonify
from chatbot import load_vectorstore, setup_qa_chain

app = Flask(__name__)

vectorstore = load_vectorstore()
qa_chain = setup_qa_chain(vectorstore)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")

        if not user_input:
            return jsonify({"error": "Message not provided"}), 400

        response = qa_chain.run(user_input)

        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
