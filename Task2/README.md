# EE471 Week 10 - Task 2

This project implements the Task 1 Hugging Face pipeline tasks in an OOP module and exposes them with a Gradio front end.

The implementation follows the Hugging Face LLM Course chapter 2 idea that a pipeline combines preprocessing, model inference, and postprocessing.

## Install Dependencies

From this folder:

```powershell
..\..\.conda\python.exe -m pip install -r requirements.txt
```

## Run The Gradio App

```powershell
..\..\.conda\python.exe app.py
```

Then open the local URL printed by Gradio.

## Files

- `hf_tasks.py`: OOP module with one class for all tasks.
- `app.py`: Gradio user interface with one tab for each task.
- `requirements.txt`: Python dependencies.
