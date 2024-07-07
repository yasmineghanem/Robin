const Parser = require("tree-sitter");
const Python = require("tree-sitter-python");
const path = require("path");
const cors = require("cors");
const express = require("express");
const { FunctionDefinitionNode } = require("tree-sitter-python");
// const { Request : expressReq, Response } = require('express');
const fs = require("fs");

require("dotenv").config({
  path: path.resolve(__dirname, "../../../.env"),
}); // Load environment variables from .env

const astServer = express();
astServer.use(cors());
astServer.use(express.json());

const port = process.env.MICROSERVICE_AST_PYTHON_PORT || 3001;
const parser = new Parser();
parser.setLanguage(Python);

// Function to find a specific function node by name
function findFunctionNode(rootNode, functionName) {
  let functionNode = null;

  rootNode.children.map((node) => {
    // console.log(node)
    if (node.type === "function_definition") {
      const nameNode = node.childForFieldName("name");
      if (nameNode && nameNode.text === functionName) {
        functionNode = node;
        return false; // Stop the traversal
      }
    }
    return true; // Continue the traversal
  });

  return functionNode;
}

function appendToTree(tree, newNode) {
  return (newTree = {
    rootNode: {
      text: tree.rootNode.text + newNode.rootNode.text,
      children: [...tree.rootNode.children, ...newNode.rootNode.children],
    },
  });
}

function editTree(tree, newNode, index) {
  return (newTree = {
    rootNode: {
      text:
        tree.rootNode.text.slice(0, index) +
        newNode.rootNode.text +
        tree.rootNode.text.slice(index),
      children: [
        ...tree.rootNode.children.slice(0, index),
        ...newNode.rootNode.children,
        ...tree.rootNode.children.slice(index),
      ],
    },
  });
}

// Function to traverse and reconstruct the code without a specific node
function traverseAndRemove(node, sourceCode, nodeToRemove) {
  if (node === nodeToRemove) {
    return "";
  }

  if (node.isNamed) {
    if (node === nodeToRemove) {
      return "";
    }

    let newCode = "";
    for (let i = 0; i < node.childCount; i++) {
      newCode += traverseAndRemove(node.child(i), sourceCode, nodeToRemove);
    }
    return newCode;
  } else {
    return sourceCode.slice(node.startIndex, node.endIndex);
  }
}

// Function to recursively print the AST
function buildASTString(node, indent = 0) {
  let result =
    "  ".repeat(indent) +
    node.type +
    " (" +
    node.startPosition.row +
    ", " +
    node.startPosition.column +
    ") (" +
    node.endPosition.row +
    ", " +
    node.endPosition.column +
    ")\n";
  for (let i = 0; i < node.childCount; i++) {
    result += buildASTString(node.child(i), indent + 1);
  }
  return result;
}

// Function to recursively build the AST as a JSON object
function buildASTJson(node, code) {
  let result = {
    type: node.type,
    startPosition: node.startPosition,
    endPosition: node.endPosition,
    children: [],
  };

  if (node.type === "identifier") {
    result.name = code.slice(node.startIndex, node.endIndex);
  }

  for (let i = 0; i < node.childCount; i++) {
    result.children.push(buildASTJson(node.child(i), code));
  }

  return result;
}

// Function to recursively build a human-readable AST
function buildReadableAST(node, code) {
  let result = "";
  if (node.type === "function_declaration") {
    result += "Function Declaration: ";
    const identifierNode = node.childForFieldName("name");
    if (identifierNode) {
      result += `function ${code.slice(
        identifierNode.startIndex,
        identifierNode.endIndex
      )}`;
    }
  } else if (node.type === "call_expression") {
    result += "Function Call: ";
    const identifierNode = node.childForFieldName("function");
    if (identifierNode) {
      result += `${code.slice(
        identifierNode.startIndex,
        identifierNode.endIndex
      )}`;
    }
  } else if (node.type === "identifier") {
    result += `Identifier: ${code.slice(node.startIndex, node.endIndex)}`;
  } else if (node.type === "string") {
    result += `String: ${code.slice(node.startIndex, node.endIndex)}`;
  } else {
    result += node.type;
  }

  if (node.childCount > 0) {
    result +=
      "\n" +
      Array.from(node.children)
        .map((child) => {
          return "  " + buildReadableAST(child, code).replace(/\n/g, "\n  ");
        })
        .join("\n");
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
      console.log("aahi");
      result.name = code.slice(node.startIndex, node.endIndex);

      // result.description = `${node.type}`;
    }
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

  fs.writeFileSync(
    path.resolve(__dirname, "ast.json"),
    JSON.stringify(buildReadableASTJSON(rootNode, code))
  );

  return {
    ast: rootNode.children.toString(),
  };
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

    // return res.status(200).json({ ast: ast.rootNode.toString() });
    return res.status(200).json({ ast });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
});

astServer.listen(port, () => {
  console.log(`AST microservice listening on port ${port}`);
});
