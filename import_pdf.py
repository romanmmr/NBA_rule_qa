import os

def get_pdf_path(pdf_name):
    '''
    Inputs the pdf file name
    :return: the path to the pdf file
    '''
    my_path = os.getcwd()
    pdf_file_path = os.path.join(my_path, 'PDF_file')
    # pdf_name = os.listdir(pdf_file_path)[0]
    return os.path.join(pdf_file_path, pdf_name)