import React, { useState } from "react";

const App = () => 
{
    const [inputMessage, setInputMessage] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => 
    {
        e.preventDefault();
        setLoading(true);
        setResult(null); // Clear previous result before submitting
        try 
        {
            const response = await fetch("https://nlpassignment.onrender.com/predict", 
            {
                method: "POST",
                headers: 
                {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: inputMessage }),
            });

            const data = await response.json();
            setResult(data);
        } 
        catch (error) 
        {
            console.error("Error:", error);
            setResult({ error: "Failed to fetch prediction" });
        }
        setLoading(false);
    };

    return (
        <div style={{ textAlign: "center", padding: "40px", fontFamily: "Arial, sans-serif" }}>
            <h1 style={{ marginBottom: "40px" }}>Message Spam/Not Spam Classifier</h1>

            <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
                <input
                    type="text"
                    placeholder="Enter your message here"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    style={{
                        padding: "12px",
                        width: "400px",
                        borderRadius: "5px",
                        border: "2px solid #ccc",
                        marginRight: "10px",
                        fontSize: "16px",
                    }}
                />
                <button
                    type="submit"
                    style={{
                        padding: "12px 25px",
                        borderRadius: "5px",
                        backgroundColor: "#4CAF50",
                        color: "white",
                        border: "none",
                        fontSize: "16px",
                        cursor: "pointer",
                    }}
                    disabled={loading}
                >
                    {loading ? "Loading..." : "Submit"}
                </button>
            </form>

            {loading && <p>Processing your message...</p>}

            {result && (
              <div style={{
                  backgroundColor: "#f9f9f9",
                  padding: "20px",
                  borderRadius: "10px",
                  display: "inline-block",
                  textAlign: "left",
                  boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
                  margin: "20px auto", 
                  width: "fit-content",
              }}>
                  
                  <h3 style={{ color: "#333" }}>Message:</h3>
                  <center>
                  <p style={{ fontStyle: "italic", color: "#555" }}>{result.message}</p>
                  </center>
                  <h3 style={{ color: "#333" }}>Prediction:</h3>
                  <center>
                  <p style={{ fontWeight: "bold", fontSize: "18px", color: result.prediction === "Spam" ? "#d9534f" : "#5cb85c" }}>
                      {result.prediction}
                  </p>
                  </center>
                 
              </div>
          )}
        </div>
    );
};

export default App;