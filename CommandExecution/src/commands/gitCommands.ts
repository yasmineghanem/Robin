import * as vscode from "vscode";
import {
  GIT_DISCARD,
  GIT_PULL,
  GIT_PUSH,
  GIT_STAGE,
  GIT_STASH,
  NO_GIT_REPO,
} from "../constants/GIT";

// check if there's a repository
const hasRepository = (api: any) => {
  if (!api.repositories.length) {
    vscode.window.showErrorMessage(NO_GIT_REPO);
    return false;
  }
  return true;
};

// push to git
const gitPush = async () =>
  vscode.commands.registerCommand(GIT_PUSH, async (args) => {
    const gitExtension =
      vscode.extensions.getExtension<any>("vscode.git")?.exports;
    if (gitExtension) {
      try {
        const api = await gitExtension.getAPI(1);

        // check if there's a repository
        if (!hasRepository(api)) {
          return {
            success: false,
            message: NO_GIT_REPO,
          };
        }

        const repo = api.repositories[0];

        //Get all changes for first repository in list
        const changes = await repo.diffWithHEAD();

        // if no changes
        if (changes.length === 0) {
          vscode.window.showInformationMessage("No changes to push.");
          return {
            success: true,
            message: "No changes to push.",
          };
        }
        // stage changes
        await repo.add([]);

        // Commit changes
        await repo.commit(args?.message ?? "Robin commit");

        // Push changes
        await repo.push();

        vscode.window.showInformationMessage("Changes pushed successfully.");

        return {
          success: true,
          message: "Changes pushed successfully.",
        };
      } catch (err) {
        vscode.window.showErrorMessage("Error pushing changes.");
        console.log("ROBIN GIT", err);
        return {
          success: false,
          message: "Error pushing changes.",
        };
      }
    } else {
      vscode.window.showErrorMessage("Git extension not found.");
      return {
        success: false,
        message: "Git extension not found.",
      };
    }
  });

// pull
const gitPull = async () =>
  vscode.commands.registerCommand(GIT_PULL, async () => {
    const gitExtension =
      vscode.extensions.getExtension<any>("vscode.git")?.exports;
    if (gitExtension) {
      try {
        const api = gitExtension.getAPI(1);
        // check if there's a repository
        if (!hasRepository(api)) {
          return {
            success: false,
            message: NO_GIT_REPO,
          };
        }

        const repo = api.repositories[0];

        // Pull changes
        await repo.pull();

        vscode.window.showInformationMessage("Changes pulled successfully.");

        return {
          success: true,
          message: "Changes pulled successfully.",
        };
      } catch (err) {
        vscode.window.showErrorMessage("Error pulling changes.");
        console.log("ROBIN GIT", err);
        return {
          success: false,
          message: "Error pulling changes.",
        };
      }
    } else {
      vscode.window.showErrorMessage("Git extension not found.");
      return {
        success: false,
        message: "Git extension not found.",
      };
    }
  });

//stage changes
const gitStage = async () =>
  vscode.commands.registerCommand(GIT_STAGE, async () => {
    const gitExtension =
      vscode.extensions.getExtension<any>("vscode.git")?.exports;
    if (gitExtension) {
      try {
        const api = gitExtension.getAPI(1);
        // check if there's a repository
        if (!hasRepository(api)) {
          return {
            success: false,
            message: NO_GIT_REPO,
          };
        }

        const repo = api.repositories[0];
        //Get all changes for first repository in list
        const changes = await repo.diffWithHEAD();

        // if no changes
        if (changes.length === 0) {
          vscode.window.showInformationMessage("No present changes.");
          return {
            success: true,
            message: "No present changes .",
          };
        }

        // stage changes
        await repo.add([]);

        vscode.window.showInformationMessage("Changes staged successfully.");

        return {
          success: true,
          message: "Changes staged successfully.",
        };
      } catch (err) {
        vscode.window.showErrorMessage("Error staging changes.");
        console.log("ROBIN GIT", err);
        return {
          success: false,
          message: "Error staging changes.",
        };
      }
    } else {
      vscode.window.showErrorMessage("Git extension not found.");
      return {
        success: false,
        message: "Git extension not found.",
      };
    }
  });

//stash changes
const gitStash = async () =>
  vscode.commands.registerCommand(GIT_STASH, async () => {
    const gitExtension =
      vscode.extensions.getExtension<any>("vscode.git")?.exports;
    if (gitExtension) {
      try {
        const api = gitExtension.getAPI(1);
        // check if there's a repository
        if (!hasRepository(api)) {
          return {
            success: false,
            message: NO_GIT_REPO,
          };
        }

        const repo = api.repositories[0];

        // if no changes
        //Get all changes for first repository in list
        const changes = await repo.diffWithHEAD();

        if (changes.length === 0) {
          vscode.window.showInformationMessage("No present changes.");
          return {
            success: true,
            message: "No present changes .",
          };
        }

        // stash changes
        // await repo.createStash();
        await repo.clean(repo.diffWithHEAD(), {
          cleanAfter: true,
          force: true,
        });
        // await repo.stash();

        vscode.window.showInformationMessage("Changes stashed successfully.");

        return {
          success: true,
          message: "Changes stashed successfully.",
        };
      } catch (err) {
        vscode.window.showErrorMessage("Error stashing changes.");
        console.log("ROBIN GIT", err);
        return {
          success: false,
          message: "Error stashing changes.",
        };
      }
    } else {
      vscode.window.showErrorMessage("Git extension not found.");
      return {
        success: false,
        message: "Git extension not found.",
      };
    }
  });

//discard changes
const gitDiscard = async () =>
  vscode.commands.registerCommand(GIT_DISCARD, async () => {
    const gitExtension =
      vscode.extensions.getExtension<any>("vscode.git")?.exports;
    if (gitExtension) {
      try {
        const api = gitExtension.getAPI(1);
        // check if there's a repository
        if (!hasRepository(api)) {
          return {
            success: false,
            message: NO_GIT_REPO,
          };
        }

        const repo = api.repositories[0];

        //Get all changes for first repository in list
        const changes = await repo.diffWithHEAD();

        // if no changes
        if (changes.length === 0) {
          vscode.window.showInformationMessage("No present changes.");
          return {
            success: true,
            message: "No present changes.",
          };
        }

        //checkout .
        await repo.checkout(".");

        vscode.window.showInformationMessage("Changes discarded successfully.");

        return {
          success: true,
          message: "Changes discarded successfully.",
        };
      } catch (err) {
        vscode.window.showErrorMessage("Error discarding changes.");
        console.log("ROBIN GIT", err);
        return {
          success: false,
          message: "Error discarding changes.",
        };
      }
    } else {
      vscode.window.showErrorMessage("Git extension not found.");
      return {
        success: false,
        message: "Git extension not found.",
      };
    }
  });

// register commands
const registerGITCommands = () => {
  const commands = [gitPull, gitPush, gitStage, gitStash, gitDiscard];

  commands.forEach((command) => command());
};

export default registerGITCommands;
