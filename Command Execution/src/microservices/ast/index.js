const Parser = require("tree-sitter");
const Python = require("tree-sitter-python");
const path = require("path");
const cors = require("cors");
const express = require("express");
// const { Request : expressReq, Response } = require('express');

require("dotenv").config({
  path: path.resolve(__dirname, "../../../.env"),
}); // Load environment variables from .env

const astServer = express();
astServer.use(cors());
astServer.use(express.json());

const port = process.env.MICROSERVICE_AST_PYTHON_PORT || 3001;
const parser = new Parser();
parser.setLanguage(Python);

function parseCode(code) {
  const tree = parser.parse(code);
  return tree.rootNode;
}

astServer.post("/ast", (req, res) => {
  // console.log(req);
  const code = req.body.code;

  if (!code) {
    return res.status(400).json({ error: "Code is required" });
  }

  try {
    const ast = parseCode(code);
    return res.status(200).json({ ast });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
});

astServer.listen(port, () => {
  console.log(`AST microservice listening on port ${port}`);
});
