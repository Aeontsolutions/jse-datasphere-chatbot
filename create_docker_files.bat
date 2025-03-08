@echo off
echo Creating Dockerfile and configuration files...

:: Create Dockerfile
echo FROM python:3.10-slim > Dockerfile
echo. >> Dockerfile
echo WORKDIR /app >> Dockerfile
echo. >> Dockerfile
echo # Install dependencies >> Dockerfile
echo COPY requirements.txt . >> Dockerfile
echo RUN pip install --no-cache-dir -r requirements.txt >> Dockerfile
echo. >> Dockerfile
echo # Copy source code >> Dockerfile
echo COPY . . >> Dockerfile
echo. >> Dockerfile
echo # Expose port for Streamlit >> Dockerfile
echo EXPOSE 8501 >> Dockerfile
echo. >> Dockerfile
echo CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] >> Dockerfile

:: Create requirements.txt
echo streamlit>=1.31.0 > requirements.txt
echo boto3>=1.34.0 >> requirements.txt
echo python-dotenv>=1.0.0 >> requirements.txt
echo google-auth>=2.23.0 >> requirements.txt
echo google-cloud-aiplatform>=1.36.0 >> requirements.txt
echo google-cloud-storage>=2.13.0 >> requirements.txt
echo PyPDF2>=3.0.0 >> requirements.txt
echo vertexai>=0.0.1 >> requirements.txt

echo All files created successfully!