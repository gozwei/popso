NAME pwlnet_tiny_1
ROWS
 N  OBJ
 E  conv0   
 E  out0    
 E  conv1   
 E  out1    
 E  conv2   
 E  out2    
 G  dem0    
 G  dem1    
 G  dem2    
 G  dem3    
 L  closed0 
 L  closed1 
 L  closed2 
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    y0        OBJ       51
    y1        OBJ       75
    y2        OBJ       100
    MARKER    'MARKER'                 'INTEND'
    x0_0      OBJ       12
    x0_0      out0      1
    x0_0      dem0      1
    x0_0      closed0   1
    x0_1      OBJ       4
    x0_1      out0      1
    x0_1      dem1      1
    x0_1      closed0   1
    x0_2      OBJ       13
    x0_2      out0      1
    x0_2      dem2      1
    x0_2      closed0   1
    x0_3      OBJ       16
    x0_3      out0      1
    x0_3      dem3      1
    x0_3      closed0   1
    x1_0      OBJ       6
    x1_0      out1      1
    x1_0      dem0      1
    x1_0      closed1   1
    x1_1      OBJ       19
    x1_1      out1      1
    x1_1      dem1      1
    x1_1      closed1   1
    x1_2      OBJ       8
    x1_2      out1      1
    x1_2      dem2      1
    x1_2      closed1   1
    x1_3      OBJ       18
    x1_3      out1      1
    x1_3      dem3      1
    x1_3      closed1   1
    x2_0      OBJ       2
    x2_0      out2      1
    x2_0      dem0      1
    x2_0      closed2   1
    x2_1      OBJ       15
    x2_1      out2      1
    x2_1      dem1      1
    x2_1      closed2   1
    x2_2      OBJ       18
    x2_2      out2      1
    x2_2      dem2      1
    x2_2      closed2   1
    x2_3      OBJ       19
    x2_3      out2      1
    x2_3      dem3      1
    x2_3      closed2   1
    l0_0      conv0     1
    l0_1      OBJ       87.157
    l0_1      conv0     1
    l0_1      out0      -11.75
    l0_2      OBJ       158.328
    l0_2      conv0     1
    l0_2      out0      -23.5
    l0_3      OBJ       208.75
    l0_3      conv0     1
    l0_3      out0      -35.25
    l0_4      OBJ       243.963
    l0_4      conv0     1
    l0_4      out0      -47
    l1_0      conv1     1
    l1_1      OBJ       170.593
    l1_1      conv1     1
    l1_1      out1      -18.75
    l1_2      OBJ       310.414
    l1_2      conv1     1
    l1_2      out1      -37.5
    l1_3      OBJ       422.819
    l1_3      conv1     1
    l1_3      out1      -56.25
    l1_4      OBJ       500.609
    l1_4      conv1     1
    l1_4      out1      -75
    l2_0      conv2     1
    l2_1      OBJ       182.763
    l2_1      conv2     1
    l2_1      out2      -16.75
    l2_2      OBJ       364.483
    l2_2      conv2     1
    l2_2      out2      -33.5
    l2_3      OBJ       447.487
    l2_3      conv2     1
    l2_3      out2      -50.25
    l2_4      OBJ       528.468
    l2_4      conv2     1
    l2_4      out2      -67
RHS
    RHS1      conv0     1
    RHS1      conv1     1
    RHS1      conv2     1
    RHS1      dem0      33
    RHS1      dem1      24
    RHS1      dem2      16
    RHS1      dem3      19
BOUNDS
 BV BND1      y0      
 BV BND1      y1      
 BV BND1      y2      
 LO BND1      x0_0      11
 SC BND1      x0_0      33
 UP BND1      x0_1      24
 UP BND1      x0_2      16
 UP BND1      x0_3      19
 LO BND1      x1_0      11
 SC BND1      x1_0      33
 UP BND1      x1_1      24
 LO BND1      x1_2      5
 SC BND1      x1_2      16
 UP BND1      x1_3      19
 UP BND1      x2_0      33
 UP BND1      x2_1      24
 UP BND1      x2_2      16
 UP BND1      x2_3      19
 UP BND1      l0_0      1
 UP BND1      l0_1      1
 UP BND1      l0_2      1
 UP BND1      l0_3      1
 UP BND1      l0_4      1
 UP BND1      l1_0      1
 UP BND1      l1_1      1
 UP BND1      l1_2      1
 UP BND1      l1_3      1
 UP BND1      l1_4      1
 UP BND1      l2_0      1
 UP BND1      l2_1      1
 UP BND1      l2_2      1
 UP BND1      l2_3      1
 UP BND1      l2_4      1
SOS
 S2 SOS  pwl0  1
    l0_0      1
    l0_1      2
    l0_2      3
    l0_3      4
    l0_4      5
 S2 SOS  pwl1  1
    l1_0      1
    l1_1      2
    l1_2      3
    l1_3      4
    l1_4      5
 S2 SOS  pwl2  1
    l2_0      1
    l2_1      2
    l2_2      3
    l2_3      4
    l2_4      5
 S1 SOS  single2  1
    x0_2      1
    x1_2      2
    x2_2      3
INDICATORS
 IF  closed0   y0        0
 IF  closed1   y1        0
 IF  closed2   y2        0
ENDATA
