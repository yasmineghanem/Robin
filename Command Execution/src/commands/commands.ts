import * as vscode from "vscode";
import registerFileSystemCommands from "./fileSystemCommands";
import registerIDECommands from "./IDECommands";
import registerGITCommands from "./gitCommands";
import registerCodeCommands from "./codeCommands";

const activateRobin = () =>
  vscode.commands.registerCommand("robin.activate", () => {
    vscode.window.showInformationMessage("Robin Activated!");
  });
// register commands
const registerAllCommands = () => {
  activateRobin();
  registerFileSystemCommands();
  registerIDECommands();
  registerCodeCommands();
  registerGITCommands();
};
export default registerAllCommands;
