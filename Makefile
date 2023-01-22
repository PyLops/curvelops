watchdoc:
	while inotifywait -q -r curvelops/ -e create,delete,modify; do { make doc; }; done

doc:
	cd docs && rm -rf build && sphinx-apidoc -f -M -o source/ ../curvelops && make html && cd -
	# Add -P to sphinx-apidoc to include private files

servedoc:
	python -m http.server --directory docs/build/html/
