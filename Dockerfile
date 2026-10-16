FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 50051
CMD ["python", "grpc_server.py"]
