npm i bootstrap bootstrap-vue axios
npm install -g serve  
npm run build --prefix pdf-upload-app  

uvicorn app.main:app --reload --port 8002
serve -s pdf-upload-app/dist