run-db:
	docker run --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=1234 -e POSTGRES_DB=postgres-db -v ${PWD}/db_data:/var/lib/postgresql/data -d postgres:14.1
