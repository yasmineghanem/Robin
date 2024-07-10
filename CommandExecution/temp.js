const axios = require("axios");

const url = "http://localhost:3000/declare-var";

const requestData = {
  operation: null,
  parameters: [
    {
      name: "ahmed",
      type: null,
      default: "asd", // assuming 'x' is a placeholder, replace it with your actual default value
    },
  ],
};

axios
  .post(url, requestData)
  .then((response) => {
    console.log("Server response:", response.data);
  })
  .catch((error) => {
    console.error("Error making the request:", error.message);
  });
