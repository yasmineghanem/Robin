const Parser = require("tree-sitter");
const Python = require("tree-sitter-python");
const path = require("path");
const cors = require("cors");
const express = require("express");
const { FunctionDefinitionNode } = require("tree-sitter-python");
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

function parseCode(code) {
  const tree = parser.parse(code);
  const rootNode = tree.rootNode;

  // Find the function node
  // const functionNode = findFunctionNode(rootNode, "test_func");
  // console.log(functionNode.text);

  // add a new variable declaration
  const newVariable = "v = 5";

  const newVariableNode = parser.parse(newVariable);

  console.log("-----------------------------------------");
  console.log(newVariableNode.children);
  console.log(rootNode.children);
  console.log(rootNode.endPosition);

  console.log("-----------------------------------------");

  // add the new variable to the root node
  // tree.edit({
  //   startIndex: rootNode.endIndex,
  //   oldEndIndex: rootNode.endIndex,
  //   newEndIndex: rootNode.endIndex + newVariableNode.endIndex,
  //   startPosition: {
  //     row: rootNode.endPosition.row + 1,
  //     column: rootNode.endPosition.column,
  //   },
  //   oldEndPosition: {
  //     row: rootNode.endPosition.row,
  //     column: rootNode.endPosition.column,
  //   },
  //   newEndPosition: {
  //     row: rootNode.endPosition.row + newVariableNode.endIndex,
  //     column: rootNode.endPosition.column + newVariableNode.endIndex,
  //   },
  // });

  // tree.edit({
  //   startIndex: 0,
  //   oldEndIndex: tree.rootNode.endIndex,
  //   newEndIndex: tree.rootNode.endIndex + newVariableNode.rootNode.endIndex,
  //   startPosition: { row: 0, column: 0 },
  //   oldEndPosition: tree.rootNode.endPosition,
  //   newEndPosition: {
  //     row:
  //       tree.rootNode.endPosition.row +
  //       newVariableNode.rootNode.endPosition.row,
  //     column:
  //       tree.rootNode.endPosition.column +
  //       newVariableNode.rootNode.endPosition.column,
  //   },
  // });

  // console.log(tree.rootNode.dot);

  // split the source code on \n
  // const codeLines = code.split("\n");

  // const newTree = parser.parse((index, position) => {
  //   let line = codeLines[position.row];
  //   if (line) {
  //     return line.slice(position.column);
  //   }
  // });

  const nodeToRemove = rootNode.namedChildren[1];

  // Reconstruct the source code without the node
  const newSourceCode = new FunctionDefinitionNode({
    
  });
  
  

  // Print the modified source code
  console.log(newSourceCode);

  // Reparse the modified source code
  const newTree = parser.parse(newSourceCode);
  const newRootNode = newTree.rootNode;

  // // new tree that consists of the old tree and the new variable code node
  // // const newTree = appendToTree(tree, newVariableNode);
  // const newTree = editTree(tree, newVariableNode,0);
  console.log(newTree.rootNode);

  return {};
}

astServer.post("/ast", (req, res) => {
  // console.log(req);
  const code = req.body.code;

  if (!code) {
    return res.status(400).json({ error: "Code is required" });
  }

  try {
    const ast = parseCode(code);
    // return res.status(200).json({ ast: ast.rootNode.toString() });
    return res.status(200).json({ ast });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
});

astServer.listen(port, () => {
  console.log(`AST microservice listening on port ${port}`);
});
