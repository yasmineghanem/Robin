import request from "supertest";
import express from "express";
import router from "../../communication/IDERouter";
import vscode from "vscode";

// Mock the vscode commands
jest.mock("vscode", () => ({
  commands: {
    executeCommand: jest.fn().mockResolvedValue({ success: true }),
  },
}));

const app = express();
app.use(express.json());
app.use("/", router);

describe("VSCode Commands API", () => {
  it("should navigate to line", async () => {
    const response = await request(app).post("/go-to-line").send({ line: 10 });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Navigated to line!" });
  });

  it("should navigate to file", async () => {
    const response = await request(app)
      .post("/go-to-file")
      .send({ filePath: "some/file/path" });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Navigated to file!" });
  });

  it("should focus terminal", async () => {
    const response = await request(app).get("/focus-terminal");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Terminal focused!" });
  });

  it("should create new terminal", async () => {
    const response = await request(app).get("/new-terminal");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "New terminal created!" });
  });

  it("should kill terminal", async () => {
    const response = await request(app).get("/kill-terminal");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Terminal killed!" });
  });

  it("should copy", async () => {
    const response = await request(app).get("/copy");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Copied!" });
  });

  it("should paste", async () => {
    const response = await request(app).get("/paste");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Pasted!" });
  });

  it("should cut", async () => {
    const response = await request(app).get("/cut");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Cut!" });
  });

  it("should undo", async () => {
    const response = await request(app).get("/undo");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Undo Done!" });
  });

  it("should redo", async () => {
    const response = await request(app).get("/redo");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Redo Done!" });
  });

  it("should select kernel", async () => {
    const response = await request(app)
      .post("/select-kernel")
      .send({ kernel: "Python 3" });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Kernel selected!" });
  });

  it("should run notebook cell", async () => {
    const response = await request(app).get("/run-notebook-cell");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Notebook cell run!" });
  });

  it("should run all notebook cells", async () => {
    const response = await request(app).get("/run-notebook");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Notebook run!" });
  });

  it("should run python file", async () => {
    const response = await request(app)
      .post("/run-python-file")
      .send({ filePath: "some/file/path.py" });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Python file run!" });
  });

  it("should select", async () => {
    const response = await request(app)
      .get("/select")
      .send({ range: { start: 0, end: 5 } });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Selected!" });
  });

  it("should select range", async () => {
    const response = await request(app)
      .get("/select-range")
      .send({ range: { start: 0, end: 10 } });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Selected Range!" });
  });

  it("should find", async () => {
    const response = await request(app).get("/find");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({ message: "Find!" });
  });
});
