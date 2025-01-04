class ErrorHandler(object):
    def __new__(err_handler):
        if not hasattr(err_handler, 'instance'):
            err_handler.instance = super(ErrorHandler, err_handler).__new__(err_handler)
        return err_handler.instance