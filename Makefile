IMAGE ?= <account>.dkr.ecr.<region>.amazonaws.com/<repository>
TAG ?= latest
SERVICE_NAME ?= fraud-app
PORT ?= 8080

build:
	docker build -t $(IMAGE):$(TAG) .

push:
	docker push $(IMAGE):$(TAG)

run:
	docker run -d --name $(SERVICE_NAME) -p $(PORT):$(PORT) --restart unless-stopped $(IMAGE):$(TAG)

pull-run:
	docker pull $(IMAGE):latest
	docker stop $(SERVICE_NAME) || true
	docker rm $(SERVICE_NAME) || true
	docker run -d --name $(SERVICE_NAME) -p $(PORT):$(PORT) --restart unless-stopped $(IMAGE):latest

.PHONY: build push run pull-run
