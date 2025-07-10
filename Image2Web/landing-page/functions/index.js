const functions = require("firebase-functions");
const admin = require("firebase-admin");
const axios = require("axios");
const { Configuration, OpenAI } = require("openai");

admin.initializeApp();
const db = admin.firestore();

// Initialize OpenAI (replace your API key)
const openai = new OpenAI({
  apiKey: "YOUR_OPENAI_API_KEY",
});

exports.generateCodeFromWireframe = functions.firestore
  .document("wireframes/{docId}")
  .onCreate(async (snap, context) => {
    const data = snap.data();
    const fileURL = data.fileURL;
    const description = data.description;
    const model = data.model || "gpt-4o";

    try {
      // Sample logic: Get image as base64 if needed
      // const response = await axios.get(fileURL, { responseType: "arraybuffer" });
      // const imageBase64 = Buffer.from(response.data, "binary").toString("base64");

      // Call AI model with image URL + description
      const aiResponse = await openai.chat.completions.create({
        model: model,
        messages: [
          {
            role: "system",
            content: "You are a helpful AI that converts wireframes into HTML/CSS/JS code."
          },
          {
            role: "user",
            content: `Here is the wireframe: ${fileURL}. Description: ${description}. Please generate clean, responsive HTML/CSS/JS code for this wireframe.`
          }
        ],
        max_tokens: 4000,
      });

      const generatedCode = aiResponse.choices[0].message.content;

      // Update document with generated code
      await snap.ref.update({
        generatedCode: generatedCode,
        status: "completed",
      });

      console.log("Code generation successful for doc:", context.params.docId);
    } catch (error) {
      console.error("Error generating code:", error);
      await snap.ref.update({
        status: "error",
        errorMessage: error.message,
      });
    }
  });
