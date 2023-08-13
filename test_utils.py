import numpy as np
import openpyxl

def matrix_symmetric_test(matrix):
    # matrix is symmetric if elements are same as the transpose
    return np.allclose(matrix, matrix.T)


def tensor_symmetric_test(tensor):
    results = []
    for m in tensor:
        results.append(matrix_symmetric_test(m))

    return all(results)


def write_to_excel(datalist, n_clusters):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for element in datalist:
        sheet.append([element])

    excel_filename = 'result_' + str(n_clusters) + '_clusters.xlsx'
    workbook.save(excel_filename)
