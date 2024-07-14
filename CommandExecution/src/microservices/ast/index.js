const Parser = require("tree-sitter");
const Python = require("tree-sitter-python");
const path = require("path");
const cors = require("cors");
const express = require("express");
const { FunctionDefinitionNode } = require("tree-sitter-python");
// const { Request : expressReq, Response } = require('express');
const fs = require("fs");
const e = require("cors");

require("dotenv").config({
  path: path.resolve(__dirname, "../../../.env"),
}); // Load environment variables from .env

const astServer = express();
astServer.use(cors());
astServer.use(express.json());

const port = process.env.MICROSERVICE_AST_PYTHON_PORT || 3001;
const parser = new Parser();
parser.setLanguage(Python);


// Function to recursively build the AST as a JSON object
function buildASTJson(node, code) {
  let result = {
    type: node.type,
    startPosition: node.startPosition,
    endPosition: node.endPosition,
    children: []
  };

  for (let i = 0; i < node.childCount; i++) {
    result.children.push(buildASTJson(node.child(i), code));
  }

  return result;
}


// Function to recursively build a human-readable AST in JSON format
function buildReadableASTJSON(node, code) {
  // check if any key has start or end
  // if (!node.childCount) {
  //   return;
  // }

  let result = {
    type: node.type,
    children: [],
  };

  // check if named, then add the name in the description
  if (node.isNamed) {
    console.log(node);
    if (
      ["identifier", "integer", "float"].includes(node.type) ||
      node.type.includes("string")
    ) {
      // console.log("aahi");
      result.name = code.slice(node.startIndex, node.endIndex);
    }
  } else if (node.type.length < 3 && ![",", ".", ":",].includes(node.type)) {
    result.name = node.type;
  }

  for (let i = 0; i < node.childCount; i++) {
    if (node.type !== "string") {
      let c = buildReadableASTJSON(node.child(i), code);
      if (c) {
        result.children.push(c);
      }
    }
  }

  // if no children, remove the key
  if (!result.children.length) {
    delete result.children;
  }
  return result;
}

function parseCode(code) {
  const tree = parser.parse(code);
  const rootNode = tree.rootNode;
  // console.log(rootNode.children);
  // write children to file in relative path
  fs.writeFileSync(
    path.resolve(__dirname, "ast_bgd.json"),
    JSON.stringify(buildASTJson(rootNode, code))
  );
  // fs.writeFileSync(
  //   path.resolve(__dirname, "ast.txt"),
  //   buildReadableAST(rootNode, code)
  // );

  

  let ast_parsed = JSON.stringify(buildReadableASTJSON(rootNode, code));
  fs.writeFileSync(
    path.resolve(__dirname, "ast.json"),
    ast_parsed
  );
  // console.log(ast_parsed);
  return ast_parsed;
}

/**
 * Given a code file, if any function, or parameter has some functional documentation
 * we need to extract that functional documentation
 */

astServer.post("/ast", (req, res) => {
  // console.log(req);
  const code = req.body.code;
  // console.log(code)

  if (!code) {
    return res.status(400).json({ error: "Code is required" });
  }

  try {
    const ast = buildReadableASTJSON(parser.parse(code).rootNode, code);


    parseCode(code);

    return res.status(200).json({ ast });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
});

astServer.listen(port, () => {
  console.log(`AST microservice listening on port ${port}`);
});
