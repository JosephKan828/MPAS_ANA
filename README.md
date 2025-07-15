# AOGS -- MPAS EAPE Budget 

# EAPE Budget Equation
According to Nakamura and Takayabu (2023), the EAPE budget equation is written as:
$$
    \frac{1}{2\sigma}\alpha^{\prime 2} = \alpha^\prime\omega^\prime + \frac{R}{p C_p \sigma} \alpha^\prime Q_1^\prime
$$

where $\sigma$ is static stability defined as 
$-\frac{1}{[\rho][\theta]}\frac{\partial [\theta]}{\partial p}$, where the square bracket means the average taken in time and longitude. The prime denotes the deviation from such average value. All the other symbols follow convention in meteorology.

# Budget Analysis on different terms
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

## Conversion $\left( \alpha^\prime \omega^\prime \right)$
### CNTL
![CNTL conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Conversion/CNTL.png)

### NCRF
![NCRF conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Conversion/NCRF.png)

## $\alpha$ Variance $\left( \alpha^{\prime 2} \right)$
### CNTL
![CNTL conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Variance/CNTL.png)

### NCRF
![NCRF conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Variance/NCRF.png)

## $\alpha$ Variance Tendency $\left( \frac{1}{2\sigma} \alpha^{\prime 2} \right)$
### CNTL
![CNTL conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Tendency/CNTL.png)

### NCRF
![NCRF conversion](https://github.com/JosephKan828/MPAS_ANA/blob/main/Figure/EAPE/Tendency/NCRF.png)
