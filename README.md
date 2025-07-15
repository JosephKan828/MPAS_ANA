# AOGS -- MPAS EAPE Budget 

# EAPE Budget Equation
According to Nakamura and Takayabu (2023), the EAPE budget equation is written as:

$$\frac{1}{2\sigma}\alpha^{\prime 2} = \alpha^\prime\omega^\prime + \frac{R}{p C_p \sigma} \alpha^\prime Q_1^\prime$$

where $\sigma$ is static stability defined as 
$-\frac{1}{[\rho][\theta]}\frac{\partial [\theta]}{\partial p}$, where the square bracket means the average taken in time and longitude. The prime denotes the deviation from such average value. All the other symbols follow convention in meteorology.

# Budget Analysis on different terms
## Computation Procedure
1. Select KW events based on space-time filter with 95\% single-tailed Z-test in vertical-integrated $Q_1$.
2. Rolling events to the center of time series at specific longitude for composite, including $\alpha^\prime, \omega^\prime, Q_1^\prime$.
3. Compute profile based on equation.
4. Vertical profile (right panel for each figure) is mean over the wave period.
5. Value shown as title of right panel is the vertically integrated value of the vertical profile.
6. The difference profile is NCRF - CNTL, and the contour is the profile of CNTL

## Result Table
|Terms|Result (in modulus)|
|:---:|:---:|
|Generation| CNTL > NCRF |
|Conversion| CNTL > NCRF |
|Variance| NCRF > CNTL |
|Tendency| NCRF > CNTL |

## Generation $\left( \frac{R}{p C_p \sigma} \alpha^\prime Q_1^\prime \right)$
### CNTL
![CNTL generation](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Generation/CNTL.png)

### NCRF
![NCRF generation](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Generation/NCRF.png)

### Difference
![generation_difference](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Generation/diff.png)

## Conversion $\left( \alpha^\prime \omega^\prime \right)$
### CNTL
![CNTL conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Conversion/CNTL.png)

### NCRF
![NCRF conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Conversion/NCRF.png)

### Difference
![conversion_difference](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Conversion/diff.png)

## $\alpha$ Variance $\left( \alpha^{\prime 2} \right)$
### CNTL
![CNTL conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Variance/CNTL.png)

### NCRF
![NCRF conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Variance/NCRF.png)

### Difference
![conversion_difference](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Variance/diff.png)

## $\alpha$ Variance Tendency $\left( \frac{1}{2\sigma} \alpha^{\prime 2} \right)$
### CNTL
![CNTL conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Tendency/CNTL.png)

### NCRF
![NCRF conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Tendency/NCRF.png)

### Difference
![conversion_difference](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Tendency/diff.png)

## Generation + Conversion $\left( \frac{R}{p C_p \sigma} \alpha^\prime Q_1^\prime + \alpha^\prime \omega^\prime  \right)$
### CNTL
![CNTL conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Gen+Conv/CNTL.png)

### NCRF
![NCRF conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Gen+Conv/NCRF.png)

### Difference
![conversion_difference](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Gen+Conv/diff.png)
