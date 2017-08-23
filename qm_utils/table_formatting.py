import os
import win32com.client


QM_1_DIR = os.path.dirname(__file__)
QM_0_DIR = os.path.dirname(QM_1_DIR)
PROG_DATA_DIR = os.path.join(QM_0_DIR, 'pucker_prog_data')
COMP_CLASSES_DIR = os.path.join(PROG_DATA_DIR, 'comparison_classes')

molecules = os.listdir(COMP_CLASSES_DIR)

xl=win32com.client.Dispatch("Excel.Application")

personal_filename = "C:/Users/justi/AppData/Roaming/Microsoft/Excel/XLSTART/PERSONAL.XLSB"

for i in range(len(molecules)):
    molecule = molecules[i]
    TABLES_DIR = os.path.join(os.path.join(COMP_CLASSES_DIR, molecule), 'tables')
    tables = os.listdir(TABLES_DIR)

    for j in range(len(tables)):
        table = tables[j]
        filename = os.path.join(TABLES_DIR, table)

        xl.Workbooks.Open(filename, ReadOnly=1)

        # if 'RMSD' in table:
        #     xl.Application.Run(personal_filename + "!SaveRMSDTable")
        # elif 'pathway_weightings' in table:
        #     pass
        # elif 'weighted_gibbs' in table:
        #     xl.Application.Run(personal_filename + "!SaveGibbsTable")

        if 'RMSD' in table:
            xl.Application.Run('SaveRMSDTable')
        elif 'pathway_weightings' in table:
            pass
        elif 'weighted_gibbs' in table:
            xl.Application.Run('SaveGibbsTable')

        xl.Application.Quit()

del xl
