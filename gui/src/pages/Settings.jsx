import React from 'react'
import { 
  Box, 
  Typography, 
  Card, 
  CardContent,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListSubheader,
  Switch,
  Slider,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider
} from '@mui/material'
import { 
  SmartToy, 
  Security, 
  Storage, 
  CloudQueue, 
  VpnKey, 
  Notifications,
  NetworkCheck,
  Code,
  Email,
  ChatBubbleOutline,
  SystemUpdateAlt,
  Speed,
  BugReport,
  Backup,
  Language
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const Settings = () => {
  const [apiSettings, setApiSettings] = React.useState({
    openaiEnabled: true,
    openrouterEnabled: true,
    huggingfaceEnabled: true,
    geminiEnabled: true,
    deepseekEnabled: true
  })

  const [agentSettings, setAgentSettings] = React.useState({
    autoMode: true,
    resourceLimit: 70,
    maxTasks: 10,
    learningRate: 0.005,
    webBrowsing: true,
    emailEnabled: true,
    smsEnabled: true,
    socialEnabled: true,
    codeEditingEnabled: true
  })

  const [securitySettings, setSecuritySettings] = React.useState({
    contentFiltering: true,
    apiRateLimiting: true,
    sandboxCodeExecution: true,
    automaticBackups: true,
    backupFrequency: 'daily'
  })

  const handleToggleChange = (setting, section) => (event) => {
    if (section === 'api') {
      setApiSettings({
        ...apiSettings,
        [setting]: event.target.checked
      })
    } else if (section === 'agent') {
      setAgentSettings({
        ...agentSettings,
        [setting]: event.target.checked
      })
    } else if (section === 'security') {
      setSecuritySettings({
        ...securitySettings,
        [setting]: event.target.checked
      })
    }
  }

  const handleSliderChange = (setting) => (event, newValue) => {
    setAgentSettings({
      ...agentSettings,
      [setting]: newValue
    })
  }

  const handleSelectChange = (setting) => (event) => {
    setSecuritySettings({
      ...securitySettings,
      [setting]: event.target.value
    })
  }

  return (
    <Box sx={{ p: 3 }}>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h3" sx={{ color: 'white', mb: 4, fontWeight: 700 }}>
          ⚙️ System Settings
        </Typography>
      </motion.div>

      <Grid container spacing={3}>
        {/* AI Model Settings */}
        <Grid item xs={12} lg={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
                  AI Model Configuration
                </Typography>
                
                <List sx={{ width: '100%' }}>
                  <ListSubheader 
                    sx={{ 
                      background: 'transparent', 
                      color: '#667eea',
                      fontWeight: 600
                    }}
                  >
                    API Providers
                  </ListSubheader>
                  
                  <ListItem>
                    <ListItemIcon>
                      <VpnKey sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="OpenAI"
                      secondary="Primary text generation and analysis"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={apiSettings.openaiEnabled}
                      onChange={handleToggleChange('openaiEnabled', 'api')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <VpnKey sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="OpenRouter"
                      secondary="Router for multiple LLM providers"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={apiSettings.openrouterEnabled}
                      onChange={handleToggleChange('openrouterEnabled', 'api')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <VpnKey sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Gemini"
                      secondary="Google's multimodal model"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={apiSettings.geminiEnabled}
                      onChange={handleToggleChange('geminiEnabled', 'api')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <VpnKey sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="HuggingFace"
                      secondary="Open source model access"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={apiSettings.huggingfaceEnabled}
                      onChange={handleToggleChange('huggingfaceEnabled', 'api')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <VpnKey sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="DeepSeek"
                      secondary="Code generation specialist"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={apiSettings.deepseekEnabled}
                      onChange={handleToggleChange('deepseekEnabled', 'api')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>

                  <ListSubheader 
                    sx={{ 
                      background: 'transparent', 
                      color: '#667eea',
                      fontWeight: 600,
                      mt: 2
                    }}
                  >
                    Model Selection
                  </ListSubheader>
                  
                  <ListItem>
                    <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
                      <InputLabel id="primary-model-label" 
                        sx={{ color: '#b8c5d6', 
                        '&.Mui-focused': { color: '#667eea' } 
                      }}>
                        Primary AI Model
                      </InputLabel>
                      <Select
                        labelId="primary-model-label"
                        value="gpt-4o"
                        label="Primary AI Model"
                        sx={{ 
                          color: 'white',
                          '& .MuiOutlinedInput-notchedOutline': {
                            borderColor: 'rgba(255, 255, 255, 0.1)'
                          },
                          '&:hover .MuiOutlinedInput-notchedOutline': {
                            borderColor: 'rgba(255, 255, 255, 0.2)'
                          },
                          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                            borderColor: '#667eea'
                          }
                        }}
                      >
                        <MenuItem value="gpt-4o">OpenAI GPT-4o</MenuItem>
                        <MenuItem value="claude-3-5-sonnet">Claude 3.5 Sonnet</MenuItem>
                        <MenuItem value="gemini-2-0-flash">Gemini 2.0 Flash</MenuItem>
                        <MenuItem value="deepseek-coder">DeepSeek Coder</MenuItem>
                      </Select>
                    </FormControl>
                  </ListItem>
                  
                  <ListItem>
                    <FormControl fullWidth variant="outlined">
                      <InputLabel id="fallback-model-label" 
                        sx={{ color: '#b8c5d6', 
                        '&.Mui-focused': { color: '#667eea' } 
                      }}>
                        Fallback Model
                      </InputLabel>
                      <Select
                        labelId="fallback-model-label"
                        value="gemini-2-0-flash"
                        label="Fallback Model"
                        sx={{ 
                          color: 'white',
                          '& .MuiOutlinedInput-notchedOutline': {
                            borderColor: 'rgba(255, 255, 255, 0.1)'
                          },
                          '&:hover .MuiOutlinedInput-notchedOutline': {
                            borderColor: 'rgba(255, 255, 255, 0.2)'
                          },
                          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                            borderColor: '#667eea'
                          }
                        }}
                      >
                        <MenuItem value="gpt-4o">OpenAI GPT-4o</MenuItem>
                        <MenuItem value="claude-3-5-sonnet">Claude 3.5 Sonnet</MenuItem>
                        <MenuItem value="gemini-2-0-flash">Gemini 2.0 Flash</MenuItem>
                        <MenuItem value="deepseek-coder">DeepSeek Coder</MenuItem>
                      </Select>
                    </FormControl>
                  </ListItem>
                </List>
                
                <Box sx={{ mt: 2, p: 2, borderRadius: '12px', background: 'rgba(76, 175, 80, 0.1)' }}>
                  <Typography variant="body2" sx={{ color: '#4caf50' }}>
                    All AI models are properly configured and operational with valid API keys.
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Agent Settings */}
        <Grid item xs={12} lg={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
                  Autonomous Agent Settings
                </Typography>
                
                <List sx={{ width: '100%' }}>
                  <ListItem>
                    <ListItemIcon>
                      <SmartToy sx={{ color: '#667eea' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Autonomous Mode"
                      secondary="Agents operate independently without human approval"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={agentSettings.autoMode}
                      onChange={handleToggleChange('autoMode', 'agent')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)', my: 1 }} />

                  <ListSubheader 
                    sx={{ 
                      background: 'transparent', 
                      color: '#667eea',
                      fontWeight: 600 
                    }}
                  >
                    Performance Settings
                  </ListSubheader>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Storage sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary={`Resource Limit: ${agentSettings.resourceLimit}%`}
                      secondary="Maximum system resources agents can use"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <Box sx={{ width: '100%', px: 2 }}>
                      <Slider
                        value={agentSettings.resourceLimit}
                        onChange={handleSliderChange('resourceLimit')}
                        min={10}
                        max={100}
                        sx={{
                          color: '#4ecdc4',
                          '& .MuiSlider-thumb': {
                            height: 16,
                            width: 16
                          }
                        }}
                      />
                    </Box>
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <SmartToy sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary={`Concurrent Tasks: ${agentSettings.maxTasks}`}
                      secondary="Maximum number of tasks agents can process simultaneously"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <Box sx={{ width: '100%', px: 2 }}>
                      <Slider
                        value={agentSettings.maxTasks}
                        onChange={handleSliderChange('maxTasks')}
                        min={1}
                        max={20}
                        marks
                        step={1}
                        sx={{
                          color: '#4ecdc4',
                          '& .MuiSlider-thumb': {
                            height: 16,
                            width: 16
                          },
                          '& .MuiSlider-mark': {
                            backgroundColor: '#b8c5d6',
                            height: 4,
                            width: 4,
                            borderRadius: 2
                          }
                        }}
                      />
                    </Box>
                  </ListItem>
                  
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)', my: 1 }} />

                  <ListSubheader 
                    sx={{ 
                      background: 'transparent', 
                      color: '#667eea',
                      fontWeight: 600 
                    }}
                  >
                    Capability Settings
                  </ListSubheader>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Language sx={{ color: '#ffa726' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Web Browsing"
                      secondary="Agent can browse the web and search for information"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={agentSettings.webBrowsing}
                      onChange={handleToggleChange('webBrowsing', 'agent')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Email sx={{ color: '#ffa726' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Email Communications"
                      secondary="Agent can read and send emails"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={agentSettings.emailEnabled}
                      onChange={handleToggleChange('emailEnabled', 'agent')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <ChatBubbleOutline sx={{ color: '#ffa726' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="SMS Messaging"
                      secondary="Agent can send and receive SMS messages"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={agentSettings.smsEnabled}
                      onChange={handleToggleChange('smsEnabled', 'agent')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Code sx={{ color: '#ffa726' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Code Editing"
                      secondary="Agent can modify its own code and systems"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={agentSettings.codeEditingEnabled}
                      onChange={handleToggleChange('codeEditingEnabled', 'agent')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        {/* Security Settings */}
        <Grid item xs={12} lg={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
                  Security & Safety
                </Typography>
                
                <List sx={{ width: '100%' }}>
                  <ListItem>
                    <ListItemIcon>
                      <Security sx={{ color: '#ff6b6b' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Content Filtering"
                      secondary="Filter potentially harmful content or instructions"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={securitySettings.contentFiltering}
                      onChange={handleToggleChange('contentFiltering', 'security')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <NetworkCheck sx={{ color: '#ff6b6b' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="API Rate Limiting"
                      secondary="Limit API calls to prevent abuse and control costs"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={securitySettings.apiRateLimiting}
                      onChange={handleToggleChange('apiRateLimiting', 'security')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Code sx={{ color: '#ff6b6b' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Sandboxed Code Execution"
                      secondary="Run code in an isolated environment for safety"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={securitySettings.sandboxCodeExecution}
                      onChange={handleToggleChange('sandboxCodeExecution', 'security')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)', my: 1 }} />

                  <ListSubheader 
                    sx={{ 
                      background: 'transparent', 
                      color: '#667eea',
                      fontWeight: 600 
                    }}
                  >
                    Backup & Recovery
                  </ListSubheader>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Backup sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Automatic Backups"
                      secondary="Regularly back up system state and configurations"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      checked={securitySettings.automaticBackups}
                      onChange={handleToggleChange('automaticBackups', 'security')}
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Schedule sx={{ color: '#4ecdc4' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Backup Frequency"
                      secondary="How often to create automatic backups"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Select
                      value={securitySettings.backupFrequency}
                      onChange={handleSelectChange('backupFrequency')}
                      size="small"
                      sx={{ 
                        width: 120,
                        color: 'white',
                        '& .MuiOutlinedInput-notchedOutline': {
                          borderColor: 'rgba(255, 255, 255, 0.1)'
                        },
                        '&:hover .MuiOutlinedInput-notchedOutline': {
                          borderColor: 'rgba(255, 255, 255, 0.2)'
                        },
                        '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                          borderColor: '#667eea'
                        }
                      }}
                    >
                      <MenuItem value="hourly">Hourly</MenuItem>
                      <MenuItem value="daily">Daily</MenuItem>
                      <MenuItem value="weekly">Weekly</MenuItem>
                    </Select>
                  </ListItem>
                </List>
                
                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    variant="contained"
                    sx={{
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      borderRadius: '12px',
                      px: 4
                    }}
                  >
                    Apply Security Settings
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        
        {/* Maintenance */}
        <Grid item xs={12} lg={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h5" sx={{ color: 'white', mb: 3, fontWeight: 600 }}>
                  System Maintenance
                </Typography>
                
                <List sx={{ width: '100%' }}>
                  <ListSubheader 
                    sx={{ 
                      background: 'transparent', 
                      color: '#667eea',
                      fontWeight: 600 
                    }}
                  >
                    System Updates
                  </ListSubheader>
                  
                  <Box sx={{ 
                    p: 2, 
                    mb: 3, 
                    borderRadius: '12px', 
                    background: 'rgba(76, 175, 80, 0.1)',
                    border: '1px solid rgba(76, 175, 80, 0.3)'
                  }}>
                    <Typography variant="body2" sx={{ color: '#4caf50', display: 'flex', alignItems: 'center' }}>
                      <SystemUpdateAlt sx={{ mr: 1, fontSize: 20 }} />
                      System is up to date (Version 1.5.0)
                    </Typography>
                  </Box>
                  
                  <ListItem>
                    <ListItemIcon>
                      <BugReport sx={{ color: '#ffa726' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Debug Mode"
                      secondary="Enable detailed logging and diagnostics"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Switch 
                      sx={{ 
                        '& .MuiSwitch-switchBase.Mui-checked': {
                          color: '#667eea'
                        },
                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                          backgroundColor: '#667eea'
                        }
                      }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Speed sx={{ color: '#ffa726' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Performance Optimization"
                      secondary="Run system optimization tasks"
                      primaryTypographyProps={{ color: 'white' }}
                      secondaryTypographyProps={{ color: '#b8c5d6' }}
                    />
                    <Button
                      variant="outlined"
                      size="small"
                      sx={{
                        borderColor: '#ffa726',
                        color: '#ffa726',
                        '&:hover': {
                          borderColor: '#ffa726',
                          background: 'rgba(255, 167, 38, 0.1)'
                        }
                      }}
                    >
                      Optimize
                    </Button>
                  </ListItem>
                  
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)', my: 2 }} />
                  
                  <Grid container spacing={2} sx={{ px: 2, pb: 2 }}>
                    <Grid item xs={12} sm={6}>
                      <Button
                        fullWidth
                        variant="outlined"
                        sx={{
                          borderColor: '#ff6b6b',
                          color: '#ff6b6b',
                          '&:hover': {
                            borderColor: '#ff6b6b',
                            background: 'rgba(255, 107, 107, 0.1)'
                          }
                        }}
                      >
                        Reset All Agents
                      </Button>
                    </Grid>
                    
                    <Grid item xs={12} sm={6}>
                      <Button
                        fullWidth
                        variant="outlined"
                        sx={{
                          borderColor: '#ff6b6b',
                          color: '#ff6b6b',
                          '&:hover': {
                            borderColor: '#ff6b6b',
                            background: 'rgba(255, 107, 107, 0.1)'
                          }
                        }}
                      >
                        Factory Reset
                      </Button>
                    </Grid>
                  </Grid>
                </List>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 4 }}>
        <Button
          variant="outlined"
          sx={{
            borderColor: '#b8c5d6',
            color: '#b8c5d6',
            mr: 2,
            '&:hover': {
              borderColor: 'white',
              color: 'white'
            }
          }}
        >
          Cancel
        </Button>
        <Button
          variant="contained"
          sx={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: '12px',
            px: 4
          }}
        >
          Save All Settings
        </Button>
      </Box>
    </Box>
  )
}

export default Settings