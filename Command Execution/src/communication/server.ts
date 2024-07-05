import cors from "cors";
import express from "express";
import codeRouter from "./codeRouter";
import fileSystemRouter from "./fileSystemRouter";
import gitRouter from "./gitRouter";
import IDERouter from "./IDERouter";

const server = express();
// middleware
server.use(cors());
server.use(express.json());


server.use('/file-system', fileSystemRouter);
server.use('/ide', IDERouter);
server.use('/code', codeRouter);
server.use('/git', gitRouter);





// export the server
export default server;
