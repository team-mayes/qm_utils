import os
import win32com.client


QM_1_DIR = os.path.dirname(__file__)
QM_0_DIR = os.path.dirname(QM_1_DIR)
PY_DIR = os.path.dirname(QM_0_DIR)
CODES_DIR = os.path.dirname(PY_DIR)
RES_DIR = os.path.dirname(CODES_DIR)

BOX_DIR = os.path.join(RES_DIR, 'Box Sync')
TEAM_DIR = os.path.join(BOX_DIR, 'TeamMayes&Blue')
PROJ_DIR = os.path.join(TEAM_DIR, 'research_projects')
PM_DIR = os.path.join(PROJ_DIR, 'puckering_methods')
PAP_DIR = os.path.join(PM_DIR, 'manuscript')
JCTC_DIR = os.path.join(PAP_DIR, 'JCTC_format-JH')
FIG_DIR = os.path.join(JCTC_DIR, 'figures')

molecules = os.listdir(FIG_DIR)

xl=win32com.client.Dispatch("Excel.Application")

personal_filename = "C:/Users/justi/AppData/Roaming/Microsoft/Excel/XLSTART/PERSONAL.XLSB"
xl.Workbooks.Open(Filename=personal_filename, ReadOnly=1)

for i in range(len(molecules)):
    molecule = molecules[i]

    TABLES_DIR = os.path.join(os.path.join(FIG_DIR, molecule), 'tables')

    tables = os.listdir(TABLES_DIR)

    for j in range(len(tables)):
        table = tables[j]

        # if table.split('.')[1] == 'csv':
        filename = os.path.join(TABLES_DIR, table)
        xl.Workbooks.Open(Filename=filename, ReadOnly=1)

        if 'RMSD' in table:
            xl.Application.Run(personal_filename + "!SaveRMSDTable")
        elif 'pathway_weightings' in table:
            pass
        elif 'weighted_gibbs' in table:
            xl.Application.Run(personal_filename + "!SaveGibbsTable")

        xl.Application.Quit()

del xl
