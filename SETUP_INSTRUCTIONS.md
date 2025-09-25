# Medical Chatbot Setup Instructions

## Hugging Face Token Configuration

You need to set up your Hugging Face token to use the medical chatbot. You have two options:

### Option 1: Using Streamlit Secrets (Recommended)

1. Edit the file `.streamlit/secrets.toml`
2. Replace `your_huggingface_token_here` with your actual Hugging Face token
3. Get your token from: https://huggingface.co/settings/tokens

Example:
```toml
HF_TOKEN = "hf_your_actual_token_here"
```

### Option 2: Using Environment Variables

1. Set the environment variable `HF_TOKEN` with your Hugging Face token
2. On Windows (PowerShell):
   ```powershell
   $env:HF_TOKEN="hf_your_actual_token_here"
   ```
3. On Windows (Command Prompt):
   ```cmd
   set HF_TOKEN=hf_your_actual_token_here
   ```

## Running the Application

After setting up your token, run the application with:

```bash
streamlit run chat_bot.py
```

## Getting a Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Sign in or create an account
3. Click "New token"
4. Give it a name (e.g., "Medical Bot")
5. Select "Read" access
6. Copy the generated token
7. Use this token in your configuration
