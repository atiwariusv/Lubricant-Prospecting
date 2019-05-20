SELECT
     CAST('AF' as varchar(50))                AS [Source],
	 CAST(CustomerID as varchar(350))  as SourceSystemID, 
	 CustomerAddressKey
    ,[CustomerAddrName]
    ,[CustomerAddrAddress]
    ,[CustomerAddrCity]
    ,[CustomerAddrState]
    ,[CustomerAddrPostalCode]
    ,ProductDesc
    ,ProductCategory
    ,ProductSubCategory
    ,ProductType
    ,ProductGroup
    ,[ProductSegmentation]
    ,CASE WHEN ProductCategory LIKE '%MOB%' THEN 1 ELSE 0 END AS IsMobil
FROM [USVBIAnalytics].[DWAF].[vwFactSales] fs
WHERE ProductSegmentation = 'LUBES'
AND CustomerKey <> 6559
AND IsClosed = 0 AND IsNationalAccount = 'N'
AND ModeOfTransportCode = 'TRK'
AND WarehouseType = 'Warehouse'
AND WarehouseKey IN (2, 9, 19, 23, 37, 55)
AND InvoiceDateKey >= CONVERT(int, FORMAT(DATEADD(year, -1, GETDATE()), 'yyyyMMdd'))
AND (ProductSubCategory = 'PASSENGER CAR') -- Industrial, commercial, and passenger car segmentation
AND ProductGroup <> 'LBTY' 
AND ProductGroup NOT LIKE '%TE%'--Liberty & Terraclean exclusion