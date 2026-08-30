from app import app


if __name__ == "__main__":
    from license_client.runtime_bridge import prime_application_state

    prime_application_state(
        app_version="1.1.1.45",
    )

    app.run(
        debug=False,
        use_reloader=False,
    )
