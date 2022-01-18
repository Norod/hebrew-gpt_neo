.PHONY: run run-container gcloud-deploy

run:
	@streamlit run app.py --server.port=8080 --server.address=0.0.0.0

run-container:
	@docker build . -t "hebgptneo-xl-sl"
	@docker run -p 8080:8080 "hebgptneo-xl-sl"

gcloud-deploy:
	@gcloud app deploy app.yaml
