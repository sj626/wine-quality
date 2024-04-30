FROM amazonlinux:latest
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8
ENV JAVA_HOME /usr/local/openjdk-8
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1
WORKDIR /app

COPY . .

RUN yum update
RUN yum install python -y
RUN python -m ensurepip --upgrade
RUN pip3 install -r requirements.txt

EXPOSE 80 $ 8080

CMD ["python", "app.py"]
