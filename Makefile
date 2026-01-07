dev:
	flutter build apk --debug

prod:
	flutter build apk --release --target-platform android-arm,android-arm64,android-x64