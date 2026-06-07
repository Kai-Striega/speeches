install: node_modules

node_modules: package.json package-lock.json
	npm install

present: node_modules
	npx slidev slides.md

pdf: node_modules
	npx slidev export --with-clicks slides.md

html: node_modules
	npx slidev build slides.md

