build:
	docker-compose up -d --build

shell:
	docker-compose up -d
	docker-compose exec potter_gpt bash

run-ui:
	docker-compose exec potter_gpt python generate.py