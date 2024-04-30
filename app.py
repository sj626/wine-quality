from flask import Flask

def create_app(config_filename=''):
    app = Flask(__name__)
    with app.app_context():
        from views.importcsv import winepredict
        app.register_blueprint(winepredict)
    return app

app = create_app()


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=8080,80)
