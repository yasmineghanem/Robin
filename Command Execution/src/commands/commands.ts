import registerFileSystemCommands from './fileSystemCommands';
import registerIDECommands from './IDECommands';
import registerCodeCommands from './codeCommands';
import registerGITCommands from './gitCommands';

// register commands 
const registerAllCommands = () => {
  registerFileSystemCommands();
  registerIDECommands();
  registerCodeCommands();
  registerGITCommands();
};
export default registerAllCommands;