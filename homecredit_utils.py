import pandas as pd
import numpy as np
import featuretools as ft
import re
import lightgbm as lgb
from probatus.feature_elimination import ShapRFECV
from skopt import BayesSearchCV
from skopt.space import Real, Integer


def create_diffs_ratios(df: pd.DataFrame, ratio_diff_num_cols: list) -> None:
  diff_comb = []

  for col_i in ratio_diff_num_cols:
    for col_j in ratio_diff_num_cols:
      if col_i!=col_j and not set((col_i, col_j)) in diff_comb:
        df[f'{col_i}__{col_j}_ratio'] = df[col_i] / df[col_j]

        df[f'{col_i}__{col_j}_ratio'] = df[f'{col_i}__{col_j}_ratio']\
        .replace([np.inf, -np.inf], np.nan)

        df[f'{col_i}__{col_j}_diff'] = df[col_i] - df[col_j]
          
        diff_comb.append(set((col_i, col_j)))

  return df


def feature_engineering_and_selection(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    bureau_balance_df: pd.DataFrame,
    pos_cash_balance_df: pd.DataFrame,
    credit_card_balance_df: pd.DataFrame,
    previous_application_df: pd.DataFrame,
    installments_payments_df: pd.DataFrame,
    target_df: str='application',
    perform_feature_selection: bool=True
  ) -> pd.DataFrame:

  # EntitySet creation for deep feature synthesis
  es = ft.EntitySet(id='homecredit-data')

  es = es.add_dataframe(dataframe_name='application', dataframe=application_df, 
                        index='SK_ID_CURR')

  es = es.add_dataframe(dataframe_name='bureau', dataframe=bureau_df, 
                        index='SK_ID_BUREAU')

  es = es.add_dataframe(dataframe_name='bureau_balance', 
                        dataframe=bureau_balance_df, index='bb_id', 
                        make_index=True)

  es = es.add_dataframe(dataframe_name='pos_cash_balance',
                        dataframe=pos_cash_balance_df, index='pcb_id', 
                        make_index=True)

  es = es.add_dataframe(dataframe_name='credit_card_balance',
                        dataframe=credit_card_balance_df, index='ccb_id', 
                        make_index=True)

  es = es.add_dataframe(dataframe_name='previous_application',
                        dataframe=previous_application_df, index='SK_ID_PREV')

  es = es.add_dataframe(dataframe_name='installments_payments',
                        dataframe=installments_payments_df, index='ip_id', 
                        make_index=True)

  relationships = [
      # parent_entity, parent_variable, child_entity, child_variable
      ('application', 'SK_ID_CURR', 'bureau', 'SK_ID_CURR'),
      ('application', 'SK_ID_CURR', 'pos_cash_balance', 'SK_ID_CURR'),
      ('application', 'SK_ID_CURR', 'previous_application', 'SK_ID_CURR'),
      ('application', 'SK_ID_CURR', 'installments_payments', 'SK_ID_CURR'),
      ('application', 'SK_ID_CURR', 'credit_card_balance', 'SK_ID_CURR'),
      ('bureau', 'SK_ID_BUREAU', 'bureau_balance', 'SK_ID_BUREAU'),
      ('previous_application', 'SK_ID_PREV', 'pos_cash_balance', 'SK_ID_PREV'),
      ('previous_application', 'SK_ID_PREV', 'installments_payments', 'SK_ID_PREV'),
      ('previous_application', 'SK_ID_PREV', 'credit_card_balance', 'SK_ID_PREV')
  ]

  for pe, pv, ce, cv in relationships:
      es = es.add_relationship(pe, pv, ce, cv)

  es.to_pickle('entityset')

  # Deep feature synthesis
  ft_df, feature_defs = ft.dfs(entityset=es,
                                    target_dataframe_name=target_df,
                                    max_depth=2)

  bool_cols = ft_df.select_dtypes('bool').columns
  category_cols = ft_df.select_dtypes('category').columns
  float_int_cols = ft_df.select_dtypes(include=['float', 'int']).columns

  ft_df = pd.DataFrame(
      data=ft_df.values, columns=ft_df.columns)

  ft_df = ft_df.astype({col:'boolean' for col in bool_cols})

  ft_df = ft_df.astype({col:'category' for col in category_cols})

  ft_df = ft_df.astype({col:'Float64' for col in float_int_cols})

  ft_df = ft_df.astype({col:'float' for col in ft_df.select_dtypes(
          include=['Float64', 'boolean']).columns})

  ft_df = ft_df.astype({'TARGET':'int'})

  ft_df = ft_df.replace(['XNA', 365243.0], np.nan)

  ft_df = ft_df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

  ft_df.to_csv('ft_generated.csv', index=False)
  ft_df = pd.read_csv('ft_generated.csv')

  # Difference and ratio feature creation with selected columns
  ratio_diff_num_cols = application_df\
  .select_dtypes(exclude='object').iloc[:, 2:-74].columns.to_list()

  ft_df = create_diffs_ratios(ft_df, ratio_diff_num_cols)

  ft_df.to_csv('ft_generated_ratio_diff.csv', index=False)

  # Feature selection using probatus shap recursive feature elimination
  report = None

  if perform_feature_selection:
    ft_df = ft_df.sample(frac=0.1)

    X_train = ft_df.drop(columns='TARGET')

    y_train = ft_df['TARGET']

    lgbm_clf = lgb.LGBMClassifier(random_state=0, device_type='gpu',
                                  class_weight='balanced')

    lgb_param_grid = {'n_estimators': Integer(50, 1000),
                      'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                      'num_leaves': Integer(20, 3000),
                      'max_depth': Integer(3, 12),
                      'min_data_in_leaf': Integer(200, 10000),
                      'lambda_l1': Integer(0, 100), 'lambda_l2': Integer(0, 100),
                      'min_gain_to_split': Real(0.01, 15, prior='log-uniform'),
                      'bagging_fraction': Real(0.2, 0.95, prior='log-uniform'),
                      'bagging_freq': Integer(1, 2),
                      'feature_fraction': Real(0.2, 0.95, prior='log-uniform')}

    bayes_search = BayesSearchCV(lgbm_clf, lgb_param_grid, n_iter=10, cv=3, 
                                scoring='roc_auc')

    shap_elimination = ShapRFECV(clf=bayes_search, step=0.3, cv=3, random_state=0,
                                scoring='roc_auc', n_jobs=-1)

    report = shap_elimination.fit_compute(X_train, y_train, 
                                          check_additivity=False)
  
  return report


def preprocess_ds(df: pd.DataFrame, ratio_diff_num_cols: list=None) -> None:
  df = df.astype(
      {col:'category' for col in df.select_dtypes('object').columns})

  df = df.replace(['XNA', 365243.0], np.nan)

  if ratio_diff_num_cols is not None:
    diff_comb = []
    used_cols = []

    for col_i in ratio_diff_num_cols:
      for col_j in ratio_diff_num_cols:
        if col_i!=col_j and not set((col_i, col_j)) in diff_comb:
          df[f'{col_i}__{col_j}_ratio'] = df[col_i] / df[col_j]
          df[f'{col_i}__{col_j}_ratio'] = df[f'{col_i}__{col_j}_ratio']\
          .replace([np.inf, -np.inf], np.nan)

          df[f'{col_i}__{col_j}__diff'] = df[col_i] - df[col_j]
          
          diff_comb.append(set((col_i, col_j)))

          used_cols.append(col_i)
          used_cols.append(col_j)

    df = df.drop(columns=set(used_cols))

  return df


def create_final_features(
        application_df: pd.DataFrame,
        bureau_df: pd.DataFrame,
        bureau_balance_df: pd.DataFrame,
        pos_cash_balance_df: pd.DataFrame,
        credit_card_balance_df: pd.DataFrame,
        previous_application_df: pd.DataFrame,
        installments_payments_df: pd.DataFrame,
        final_cols: list=[
    'AMT_GOODS_PRICE__REGION_POPULATION_RELATIVE__ratio',
 'MIN__AMT_GOODS_PRICE__previous_application',
 'REGION_RATING_CLIENT__EXT_SOURCE_1__ratio',
 'AMT_ANNUITY__AMT_GOODS_PRICE__diff',
 'DAYS_BIRTH__REGION_RATING_CLIENT_W_CITY__ratio',
 'AMT_ANNUITY__REGION_POPULATION_RELATIVE__ratio',
 'AMT_CREDIT__DAYS_BIRTH__ratio',
 'AMT_ANNUITY__DAYS_BIRTH__ratio',
 'DAYS_ID_PUBLISH__HOUR_APPR_PROCESS_START__ratio',
 'LIVE_REGION_NOT_WORK_REGION__EXT_SOURCE_2__diff',
 'MIN__DAYS_DECISION__previous_application',
 'MIN__CNT_INSTALMENT_FUTURE__pos_cash_balance',
 'HOUR_APPR_PROCESS_START__EXT_SOURCE_1__ratio',
 'DAYS_BIRTH__HOUR_APPR_PROCESS_START__ratio',
 'LIVE_REGION_NOT_WORK_REGION__EXT_SOURCE_3__diff',
 'FLAG_MOBIL__EXT_SOURCE_3__diff',
 'OWN_CAR_AGE__CNT_FAM_MEMBERS__ratio',
 'AMT_INCOME_TOTAL__HOUR_APPR_PROCESS_START__ratio',
 'AMT_CREDIT__FLAG_CONT_MOBILE__ratio',
 'EXT_SOURCE_2__EXT_SOURCE_3__diff',
 'STD__DAYS_INSTALMENT__installments_payments',
 'AMT_CREDIT__DAYS_REGISTRATION__ratio',
 'REGION_RATING_CLIENT__EXT_SOURCE_1__diff',
 'STD__MONTHS_BALANCE__pos_cash_balance',
 'DAYS_EMPLOYED__EXT_SOURCE_1__ratio',
 'FLAG_CONT_MOBILE__EXT_SOURCE_1__diff',
 'DAYS_EMPLOYED__DAYS_ID_PUBLISH__ratio',
 'DAYS_BIRTH__OWN_CAR_AGE__ratio',
 'AMT_INCOME_TOTAL__EXT_SOURCE_3__ratio',
 'AMT_GOODS_PRICE__DAYS_ID_PUBLISH__ratio',
 'REGION_POPULATION_RELATIVE__CNT_FAM_MEMBERS__ratio',
 'AMT_INCOME_TOTAL__AMT_ANNUITY__ratio',
 'AMT_GOODS_PRICE__DAYS_BIRTH__diff',
 'DAYS_REGISTRATION__CNT_FAM_MEMBERS__ratio',
 'REGION_POPULATION_RELATIVE__DAYS_REGISTRATION__ratio',
 'DAYS_BIRTH__CNT_FAM_MEMBERS__ratio',
 'REGION_RATING_CLIENT__EXT_SOURCE_2__diff',
 'SKEW__AMT_INSTALMENT__installments_payments',
 'AMT_ANNUITY__DAYS_REGISTRATION__diff',
 'MEAN__AMT_INSTALMENT__installments_payments',
 'COMMONAREA_MODE',
 'AMT_GOODS_PRICE__EXT_SOURCE_1__ratio',
 'MIN__AMT_PAYMENT__installments_payments',
 'REG_CITY_NOT_WORK_CITY__EXT_SOURCE_3__diff',
 'MEAN__AMT_PAYMENT__installments_payments',
 'REGION_POPULATION_RELATIVE__REG_CITY_NOT_WORK_CITY__diff',
 'DAYS_REGISTRATION__EXT_SOURCE_3__ratio',
 'AMT_GOODS_PRICE__EXT_SOURCE_2__ratio',
 'AMT_CREDIT__AMT_GOODS_PRICE__ratio',
 'DAYS_BIRTH__EXT_SOURCE_3__diff',
 'FLAG_EMP_PHONE__EXT_SOURCE_3__ratio',
 'MIN__AMT_INSTALMENT__installments_payments',
 'AMT_INCOME_TOTAL__EXT_SOURCE_3__diff',
 'AMT_CREDIT__DAYS_ID_PUBLISH__ratio',
 'DAYS_EMPLOYED__CNT_FAM_MEMBERS__ratio',
 'REGION_RATING_CLIENT_W_CITY__EXT_SOURCE_3__ratio',
 'REGION_POPULATION_RELATIVE__EXT_SOURCE_2__diff',
 'AMT_INCOME_TOTAL__DAYS_EMPLOYED__ratio',
 'STD__NUM_INSTALMENT_NUMBER__installments_payments',
 'LANDAREA_MEDI',
 'SKEW__CNT_INSTALMENT_FUTURE__pos_cash_balance',
 'DAYS_ID_PUBLISH__CNT_FAM_MEMBERS__diff',
 'DAYS_REGISTRATION__HOUR_APPR_PROCESS_START__ratio',
 'FLAG_CONT_MOBILE__EXT_SOURCE_2__ratio',
 'MIN__MONTHS_BALANCE__pos_cash_balance',
 'AMT_ANNUITY__EXT_SOURCE_2__diff',
 'AMT_ANNUITY__REGION_RATING_CLIENT__ratio',
 'EXT_SOURCE_1__EXT_SOURCE_3__ratio',
 'AMT_CREDIT__EXT_SOURCE_1__ratio',
 'MAX__NUM_INSTALMENT_NUMBER__installments_payments',
 'AMT_GOODS_PRICE__EXT_SOURCE_3__diff',
 'REG_REGION_NOT_WORK_REGION__EXT_SOURCE_1__diff',
 'AMT_CREDIT__EXT_SOURCE_2__ratio',
 'DAYS_EMPLOYED__EXT_SOURCE_2__ratio',
 'DAYS_EMPLOYED__EXT_SOURCE_3__diff',
 'STD__AMT_PAYMENT__installments_payments',
 'MEAN__DAYS_INSTALMENT__installments_payments',
 'AMT_INCOME_TOTAL__AMT_GOODS_PRICE__ratio',
 'AMT_GOODS_PRICE__FLAG_EMP_PHONE__ratio',
 'DAYS_ID_PUBLISH__FLAG_EMP_PHONE__ratio',
 'DAYS_EMPLOYED__REGION_RATING_CLIENT_W_CITY__ratio',
 'CNT_FAM_MEMBERS__EXT_SOURCE_3__ratio',
 'NONLIVINGAREA_AVG',
 'LIVINGAREA_AVG',
 'MAX__DAYS_INSTALMENT__installments_payments',
 'COMMONAREA_AVG',
 'STD__DAYS_ENTRY_PAYMENT__installments_payments',
 'AMT_ANNUITY__DAYS_EMPLOYED__ratio',
 'HOUR_APPR_PROCESS_START__EXT_SOURCE_2__diff',
 'DAYS_EMPLOYED__REGION_RATING_CLIENT__ratio',
 'NAME_FAMILY_STATUS',
 'CNT_CHILDREN__OWN_CAR_AGE__diff',
 'YEARS_BUILD_MODE',
 'REGION_POPULATION_RELATIVE__EXT_SOURCE_2__ratio',
 'NAME_EDUCATION_TYPE',
 'DAYS_REGISTRATION__EXT_SOURCE_2__diff',
 'CNT_FAM_MEMBERS__EXT_SOURCE_1__ratio',
 'MIN__DAYS_ENTRY_PAYMENT__installments_payments',
 'AMT_CREDIT__AMT_GOODS_PRICE__diff',
 'EXT_SOURCE_3',
 'AMT_CREDIT__AMT_ANNUITY__ratio',
 'DAYS_BIRTH__DAYS_EMPLOYED__ratio',
 'CNT_CHILDREN__REGION_POPULATION_RELATIVE__diff',
 'AMT_INCOME_TOTAL__EXT_SOURCE_1__ratio',
 'CNT_FAM_MEMBERS__EXT_SOURCE_3__diff',
 'AMT_CREDIT__REGION_POPULATION_RELATIVE__ratio',
 'AMT_INCOME_TOTAL__AMT_ANNUITY__diff',
 'HOUR_APPR_PROCESS_START__EXT_SOURCE_3__diff',
 'REGION_POPULATION_RELATIVE__EXT_SOURCE_1__diff',
 'CNT_FAM_MEMBERS__HOUR_APPR_PROCESS_START__ratio',
 'CNT_CHILDREN__EXT_SOURCE_3__diff',
 'REGION_POPULATION_RELATIVE__DAYS_ID_PUBLISH__ratio',
 'REGION_RATING_CLIENT__EXT_SOURCE_3__diff',
 'OCCUPATION_TYPE',
 'SUM__CNT_INSTALMENT__pos_cash_balance',
 'FLAG_PHONE__EXT_SOURCE_3__diff',
 'DAYS_EMPLOYED__DAYS_ID_PUBLISH__diff',
 'REGION_POPULATION_RELATIVE__FLAG_EMP_PHONE__diff',
 'REGION_RATING_CLIENT_W_CITY__EXT_SOURCE_3__diff',
 'AMT_INCOME_TOTAL__DAYS_BIRTH__ratio',
 'REGION_RATING_CLIENT_W_CITY__EXT_SOURCE_2__diff',
 'DAYS_EMPLOYED__DAYS_REGISTRATION__ratio',
 'TOTALAREA_MODE',
 'MEAN__CNT_INSTALMENT__pos_cash_balance',
 'CODE_GENDER',
 'MIN__CNT_INSTALMENT__pos_cash_balance',
 'REG_CITY_NOT_LIVE_CITY__EXT_SOURCE_2__diff',
 'DAYS_BIRTH__DAYS_REGISTRATION__ratio',
 'AMT_GOODS_PRICE__DAYS_EMPLOYED__diff',
 'AMT_INCOME_TOTAL__DAYS_EMPLOYED__diff',
 'AMT_INCOME_TOTAL__EXT_SOURCE_2__ratio',
 'FLAG_CONT_MOBILE__EXT_SOURCE_3__diff',
 'BASEMENTAREA_MODE',
 'AMT_CREDIT__REGION_POPULATION_RELATIVE__diff',
 'FLAG_WORK_PHONE__EXT_SOURCE_3__diff',
 'SKEW__NUM_INSTALMENT_VERSION__installments_payments',
 'REG_CITY_NOT_LIVE_CITY__EXT_SOURCE_3__diff',
 'REGION_POPULATION_RELATIVE__EXT_SOURCE_3__diff',
 'REGION_POPULATION_RELATIVE__EXT_SOURCE_1__ratio',
 'MEAN__SELLERPLACE_AREA__previous_application',
 'AMT_CREDIT__REGION_RATING_CLIENT_W_CITY__ratio',
 'DAYS_REGISTRATION__EXT_SOURCE_3__diff',
 'SKEW__SK_DPD__pos_cash_balance',
 'REGION_POPULATION_RELATIVE__DAYS_EMPLOYED__diff',
 'CNT_FAM_MEMBERS__EXT_SOURCE_2__ratio',
 'AMT_ANNUITY__EXT_SOURCE_2__ratio',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'SKEW__DAYS_ENTRY_PAYMENT__installments_payments',
 'DAYS_BIRTH__EXT_SOURCE_1__ratio',
 'DAYS_LAST_PHONE_CHANGE',
 'REGION_RATING_CLIENT_W_CITY__EXT_SOURCE_1__ratio',
 'MAX__AMT_PAYMENT__installments_payments',
 'LIVINGAPARTMENTS_MEDI',
 'DAYS_BIRTH__FLAG_CONT_MOBILE__ratio',
 'DAYS_BIRTH__DAYS_REGISTRATION__diff',
 'STD__CNT_INSTALMENT_FUTURE__pos_cash_balance',
 'DAYS_EMPLOYED__HOUR_APPR_PROCESS_START__ratio',
 'AMT_CREDIT__DAYS_EMPLOYED__diff',
 'AMT_ANNUITY__EXT_SOURCE_3__ratio',
 'YEARS_BEGINEXPLUATATION_MEDI',
 'REGION_RATING_CLIENT__HOUR_APPR_PROCESS_START__diff',
 'LIVE_CITY_NOT_WORK_CITY__EXT_SOURCE_2__diff',
 'HOUR_APPR_PROCESS_START__EXT_SOURCE_2__ratio',
 'DAYS_BIRTH__DAYS_EMPLOYED__diff',
 'AMT_ANNUITY__DAYS_ID_PUBLISH__ratio',
 'AMT_GOODS_PRICE__DAYS_REGISTRATION__diff',
 'MIN__DAYS_INSTALMENT__installments_payments',
 'SUM__AMT_INSTALMENT__installments_payments',
 'FLAG_EMAIL__EXT_SOURCE_2__diff',
 'AMT_ANNUITY__AMT_GOODS_PRICE__ratio',
 'MEAN__MONTHS_BALANCE__pos_cash_balance',
 'DAYS_BIRTH__EXT_SOURCE_3__ratio',
 'MODE__PRODUCT_COMBINATION__previous_application',
 'DAYS_ID_PUBLISH__OWN_CAR_AGE__ratio',
 'DAYS_ID_PUBLISH__REGION_RATING_CLIENT__ratio',
 'AMT_CREDIT__EXT_SOURCE_3__ratio',
 'AMT_GOODS_PRICE__REGION_POPULATION_RELATIVE__diff',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'DAYS_EMPLOYED__EXT_SOURCE_3__ratio',
 'DAYS_EMPLOYED__FLAG_CONT_MOBILE__ratio',
 'REGION_POPULATION_RELATIVE__HOUR_APPR_PROCESS_START__ratio',
 'SUM__MONTHS_BALANCE__pos_cash_balance',
 'DAYS_REGISTRATION__REGION_RATING_CLIENT_W_CITY__ratio',
 'YEARS_BEGINEXPLUATATION_AVG',
 'AMT_INCOME_TOTAL__DAYS_ID_PUBLISH__ratio',
 'CNT_CHILDREN__EXT_SOURCE_2__diff',
 'MEAN__DAYS_CREDIT_ENDDATE__bureau',
 'DAYS_ID_PUBLISH__FLAG_CONT_MOBILE__ratio',
 'FLAG_EMAIL__EXT_SOURCE_3__diff',
 'DAYS_EMPLOYED__EXT_SOURCE_2__diff',
 'REG_REGION_NOT_LIVE_REGION__EXT_SOURCE_3__diff',
 'DAYS_ID_PUBLISH__EXT_SOURCE_1__ratio',
 'WALLSMATERIAL_MODE',
 'AMT_CREDIT__DAYS_EMPLOYED__ratio',
 'DAYS_BIRTH__DAYS_ID_PUBLISH__diff',
 'AMT_ANNUITY__CNT_FAM_MEMBERS__ratio',
 'EXT_SOURCE_1__EXT_SOURCE_2__ratio',
 'FLAG_MOBIL__EXT_SOURCE_3__ratio',
 'REGION_RATING_CLIENT__EXT_SOURCE_3__ratio',
 'SKEW__CNT_INSTALMENT__pos_cash_balance',
 'AMT_GOODS_PRICE__DAYS_BIRTH__ratio',
 'REGION_POPULATION_RELATIVE__FLAG_EMAIL__diff',
 'REGION_POPULATION_RELATIVE__OWN_CAR_AGE__diff',
 'REGION_POPULATION_RELATIVE__EXT_SOURCE_3__ratio',
 'AMT_ANNUITY__EXT_SOURCE_3__diff',
 'FLAG_PHONE__EXT_SOURCE_2__diff',
 'STD__SK_DPD__pos_cash_balance',
 'FLAG_EMP_PHONE__EXT_SOURCE_2__ratio',
 'AMT_GOODS_PRICE__DAYS_EMPLOYED__ratio',
 'AMT_ANNUITY__DAYS_EMPLOYED__diff',
 'AMT_INCOME_TOTAL__AMT_CREDIT__ratio',
 'DAYS_REGISTRATION__REGION_RATING_CLIENT__ratio',
 'DAYS_ID_PUBLISH__OWN_CAR_AGE__diff',
 'REG_CITY_NOT_LIVE_CITY__EXT_SOURCE_1__diff',
 'SKEW__MONTHS_BALANCE__pos_cash_balance',
 'AMT_GOODS_PRICE__REGION_RATING_CLIENT__ratio',
 'REGION_RATING_CLIENT__HOUR_APPR_PROCESS_START__ratio',
 'REG_CITY_NOT_WORK_CITY__EXT_SOURCE_1__diff',
 'CNT_FAM_MEMBERS__EXT_SOURCE_2__diff',
 'AMT_GOODS_PRICE__REGION_RATING_CLIENT_W_CITY__ratio',
 'FLAG_EMP_PHONE__EXT_SOURCE_1__diff',
 'LIVE_CITY_NOT_WORK_CITY__EXT_SOURCE_3__diff',
 'MAX__AMT_INSTALMENT__installments_payments',
 'AMT_CREDIT__CNT_FAM_MEMBERS__ratio',
 'AMT_ANNUITY__EXT_SOURCE_1__ratio',
 'MAX__CNT_INSTALMENT_FUTURE__pos_cash_balance',
 'NAME_INCOME_TYPE',
 'AMT_GOODS_PRICE__DAYS_ID_PUBLISH__diff',
 'AMT_GOODS_PRICE__EXT_SOURCE_2__diff',
 'WEEKDAY_APPR_PROCESS_START',
 'DAYS_ID_PUBLISH__EXT_SOURCE_2__ratio',
 'APARTMENTS_AVG',
 'REGION_RATING_CLIENT_W_CITY__EXT_SOURCE_1__diff',
 'DAYS_ID_PUBLISH__EXT_SOURCE_3__ratio',
 'AMT_ANNUITY__OWN_CAR_AGE__ratio',
 'REGION_POPULATION_RELATIVE__FLAG_PHONE__diff',
 'MEAN__CNT_INSTALMENT_FUTURE__pos_cash_balance',
 'OWN_CAR_AGE__LIVE_CITY_NOT_WORK_CITY__diff',
 'AMT_ANNUITY__FLAG_EMP_PHONE__ratio',
 'AMT_GOODS_PRICE__EXT_SOURCE_3__ratio',
 'LIVINGAREA_MODE',
 'DAYS_BIRTH__EXT_SOURCE_2__diff',
 'EXT_SOURCE_2__EXT_SOURCE_3__ratio',
 'AMT_ANNUITY__DAYS_BIRTH__diff',
 'BASEMENTAREA_MEDI',
 'HOUR_APPR_PROCESS_START__EXT_SOURCE_3__ratio',
 'SUM__AMT_PAYMENT__installments_payments',
 'MAX__MONTHS_BALANCE__pos_cash_balance',
 'OWN_CAR_AGE__EXT_SOURCE_3__ratio',
 'AMT_INCOME_TOTAL__REGION_POPULATION_RELATIVE__ratio',
 'EXT_SOURCE_2',
 'DAYS_REGISTRATION__OWN_CAR_AGE__ratio',
 'FLAG_CONT_MOBILE__EXT_SOURCE_2__diff',
 'MIN__NUM_INSTALMENT_NUMBER__installments_payments',
 'DAYS_BIRTH__DAYS_ID_PUBLISH__ratio',
 'DAYS_REGISTRATION__DAYS_ID_PUBLISH__diff',
 'REGION_POPULATION_RELATIVE__FLAG_WORK_PHONE__diff',
 'MIN__AMT_APPLICATION__previous_application',
 'FLAG_EMP_PHONE__EXT_SOURCE_2__diff',
 'BASEMENTAREA_AVG',
 'REG_REGION_NOT_LIVE_REGION__EXT_SOURCE_1__diff',
 'MIN__SELLERPLACE_AREA__previous_application',
 'LIVINGAPARTMENTS_MODE',
 'AMT_GOODS_PRICE__DAYS_REGISTRATION__ratio',
 'REGION_RATING_CLIENT__EXT_SOURCE_2__ratio',
 'DAYS_ID_PUBLISH__REGION_RATING_CLIENT_W_CITY__ratio',
 'OWN_CAR_AGE__EXT_SOURCE_2__ratio',
 'REG_REGION_NOT_LIVE_REGION__EXT_SOURCE_2__diff',
 'FLAG_EMP_PHONE__EXT_SOURCE_3__diff',
 'LIVINGAREA_MEDI',
 'DAYS_BIRTH__FLAG_EMP_PHONE__ratio',
 'FLAG_CONT_MOBILE__EXT_SOURCE_3__ratio',
 'AMT_ANNUITY__FLAG_CONT_MOBILE__ratio',
 'DAYS_ID_PUBLISH__EXT_SOURCE_1__diff',
 'REG_CITY_NOT_WORK_CITY__EXT_SOURCE_2__diff',
 'MEAN__NUM_INSTALMENT_NUMBER__installments_payments',
 'DAYS_ID_PUBLISH__EXT_SOURCE_3__diff',
 'DAYS_REGISTRATION__DAYS_ID_PUBLISH__ratio',
 'FLAG_WORK_PHONE__EXT_SOURCE_1__diff',
 'DAYS_BIRTH__REGION_RATING_CLIENT__ratio',
 'FLAG_MOBIL__EXT_SOURCE_2__ratio',
 'AMT_CREDIT__HOUR_APPR_PROCESS_START__ratio',
 'ORGANIZATION_TYPE',
 'AMT_ANNUITY__DAYS_ID_PUBLISH__diff',
 'FLAG_MOBIL__EXT_SOURCE_2__diff',
 'FLAG_MOBIL__EXT_SOURCE_1__ratio',
 'REGION_POPULATION_RELATIVE__HOUR_APPR_PROCESS_START__diff',
 'REG_REGION_NOT_WORK_REGION__EXT_SOURCE_3__diff',
 'DAYS_REGISTRATION__EXT_SOURCE_2__ratio',
 'FLAG_WORK_PHONE__EXT_SOURCE_2__diff',
 'DAYS_BIRTH__EXT_SOURCE_2__ratio',
 'REG_REGION_NOT_WORK_REGION__EXT_SOURCE_2__diff',
 'AMT_ANNUITY__HOUR_APPR_PROCESS_START__ratio',
 'AMT_ANNUITY__DAYS_REGISTRATION__ratio',
 'SKEW__AMT_PAYMENT__installments_payments',
 'AMT_CREDIT__EXT_SOURCE_3__diff',
 'AMT_GOODS_PRICE__CNT_FAM_MEMBERS__ratio',
 'FLAG_EMP_PHONE__EXT_SOURCE_1__ratio',
 'AMT_INCOME_TOTAL__DAYS_REGISTRATION__ratio',
 'MEAN__DAYS_ENTRY_PAYMENT__installments_payments',
 'REGION_RATING_CLIENT_W_CITY__EXT_SOURCE_2__ratio',
 'DAYS_ID_PUBLISH__CNT_FAM_MEMBERS__ratio',
 'REGION_POPULATION_RELATIVE__DAYS_EMPLOYED__ratio',
 'DAYS_REGISTRATION__EXT_SOURCE_1__ratio',
 'LIVE_REGION_NOT_WORK_REGION__EXT_SOURCE_1__diff',
 'STD__CNT_INSTALMENT__pos_cash_balance',
 'DAYS_REGISTRATION__FLAG_CONT_MOBILE__ratio',
 'APARTMENTS_MODE']
    ) -> pd.DataFrame:
    final_df = pd.DataFrame()

    application_df = application_df.set_index('SK_ID_CURR')

    for col in final_cols:
        splitted = col.split('__')

        if len(splitted)==1:
            final_df[col] = application_df[col]
            continue

        if splitted[0] in ['MIN', 'MEAN', 'STD', 'SKEW', 'SUM', 'MAX']:
            final_df[col] = eval(f"{splitted[2]}_df.groupby('SK_ID_CURR')['{splitted[1]}'].{splitted[0].lower()}()")
            continue

        if splitted[0]=='MODE':
            final_df[col] = eval(f"{splitted[2]}_df.groupby('SK_ID_CURR')['{splitted[1]}'].agg(lambda x: pd.Series.mode(x)[0])")
            continue

        if splitted[2]=='ratio':
            final_df[col] = application_df[splitted[0]] / application_df[splitted[1]]
            continue

        if splitted[2]=='diff':
            final_df[col] = application_df[splitted[0]] - application_df[splitted[1]]
            continue