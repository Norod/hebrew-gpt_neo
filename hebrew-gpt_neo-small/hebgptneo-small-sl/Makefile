.PHONY: run run-container gcloud-deploy

run:
	@streamlit run app.py --server.port=8080 --server.address=0.0.0.0

run-container:
	@docker build . -t "hebgptneo-small-sl"
	@docker run -p 8080:8080 "hebgptneo-small-sl"

gcloud-deploy:
	@gcloud app deploy app.yaml
