import React, { useState } from 'react'
import { 
  AppBar, 
  Toolbar, 
  IconButton, 
  Typography, 
  Box, 
  Badge, 
  Tooltip,
  Avatar,
  Menu,
  MenuItem,
  Divider
} from '@mui/material'
import { 
  NotificationsOutlined,
  SettingsOutlined,
  HelpOutlineOutlined,
  MenuOutlined,
  Brightness4Outlined,
  PersonOutlined,
  ExitToAppOutlined,
  Zap
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const Topbar = () => {
  const [anchorEl, setAnchorEl] = useState(null)
  const [notificationAnchorEl, setNotificationAnchorEl] = useState(null)
  
  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget)
  }

  const handleProfileMenuClose = () => {
    setAnchorEl(null)
  }
  
  const handleNotificationMenuOpen = (event) => {
    setNotificationAnchorEl(event.currentTarget)
  }

  const handleNotificationMenuClose = () => {
    setNotificationAnchorEl(null)
  }

  const notifications = [
    { 
      id: 1, 
      title: 'System Update', 
      description: 'Code Editor agent completed refactoring task',
      time: '10 minutes ago',
      read: false
    },
    { 
      id: 2, 
      title: 'AI Process Complete', 
      description: 'Market research analysis report is ready',
      time: '25 minutes ago',
      read: false
    },
    { 
      id: 3, 
      title: 'New Email Received', 
      description: 'Sarah Johnson sent a new project proposal',
      time: '1 hour ago',
      read: true
    }
  ]

  const unreadCount = notifications.filter(n => !n.read).length

  return (
    <AppBar 
      position="sticky"
      sx={{ 
        backgroundColor: 'transparent', 
        boxShadow: 'none',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        backgroundImage: 'none'
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton sx={{ display: { xs: 'flex', md: 'none' }, color: 'white' }}>
            <MenuOutlined />
          </IconButton>
          
          <motion.div
            animate={{ 
              scale: [1, 1.1, 1],
              opacity: [1, 0.8, 1]
            }}
            transition={{ 
              duration: 2,
              repeat: Infinity,
              repeatType: "loop"
            }}
          >
            <Zap sx={{ color: '#667eea', fontSize: 28, mr: 2, display: { xs: 'none', sm: 'block' } }} />
          </motion.div>
          
          <Typography variant="h6" sx={{ fontWeight: 700, display: { xs: 'none', sm: 'block' } }}>
            <span style={{ background: 'linear-gradient(90deg, #667eea, #764ba2)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              EBAIF Agentic AI
            </span>
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Box 
              sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                background: 'rgba(76, 175, 80, 0.2)', 
                borderRadius: '12px',
                py: 0.5,
                px: 2,
                mx: 2
              }}
            >
              <Box 
                sx={{ 
                  width: 10, 
                  height: 10, 
                  borderRadius: '50%', 
                  bgcolor: '#4caf50', 
                  mr: 1 
                }} 
              />
              <Typography 
                variant="body2" 
                sx={{ 
                  color: '#4caf50', 
                  fontWeight: 600
                }}
              >
                System Active
              </Typography>
            </Box>
          </motion.div>

          <Tooltip title="Notifications">
            <IconButton 
              sx={{ color: 'white', mx: 1 }}
              onClick={handleNotificationMenuOpen}
            >
              <Badge 
                badgeContent={unreadCount} 
                color="error"
                sx={{ '& .MuiBadge-badge': { bgcolor: '#ff6b6b' } }}
              >
                <NotificationsOutlined />
              </Badge>
            </IconButton>
          </Tooltip>

          <Menu
            anchorEl={notificationAnchorEl}
            open={Boolean(notificationAnchorEl)}
            onClose={handleNotificationMenuClose}
            PaperProps={{
              sx: {
                mt: 1.5,
                width: 320,
                background: 'rgba(26, 31, 46, 0.95)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
              }
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <Box sx={{ p: 2 }}>
              <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                Notifications
              </Typography>
              <Typography variant="body2" sx={{ color: '#b8c5d6' }}>
                System updates and alerts
              </Typography>
            </Box>
            <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
            
            {notifications.map((notification) => (
              <MenuItem 
                key={notification.id}
                onClick={handleNotificationMenuClose}
                sx={{ 
                  py: 2,
                  background: notification.read ? 'transparent' : 'rgba(102, 126, 234, 0.1)',
                  borderLeft: notification.read ? 'none' : '3px solid #667eea'
                }}
              >
                <Box sx={{ width: '100%' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography 
                      variant="subtitle2" 
                      sx={{ 
                        color: 'white', 
                        fontWeight: notification.read ? 400 : 600
                      }}
                    >
                      {notification.title}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#b8c5d6' }}>
                      {notification.time}
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ color: '#b8c5d6', mt: 0.5 }}>
                    {notification.description}
                  </Typography>
                </Box>
              </MenuItem>
            ))}
            
            <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
            <Box sx={{ p: 2, textAlign: 'center' }}>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: '#667eea',
                  cursor: 'pointer',
                  '&:hover': { textDecoration: 'underline' }
                }}
              >
                View all notifications
              </Typography>
            </Box>
          </Menu>

          <Tooltip title="Settings">
            <IconButton sx={{ color: 'white', mx: 1 }}>
              <SettingsOutlined />
            </IconButton>
          </Tooltip>

          <Tooltip title="Help">
            <IconButton sx={{ color: 'white', mx: 1 }}>
              <HelpOutlineOutlined />
            </IconButton>
          </Tooltip>

          <Tooltip title="Profile">
            <IconButton 
              onClick={handleProfileMenuOpen}
              sx={{ color: 'white', ml: 1 }}
            >
              <Avatar 
                sx={{ 
                  width: 32, 
                  height: 32,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                }}
              >
                <PersonOutlined sx={{ fontSize: 20 }} />
              </Avatar>
            </IconButton>
          </Tooltip>

          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleProfileMenuClose}
            PaperProps={{
              sx: {
                mt: 1.5,
                width: 200,
                background: 'rgba(26, 31, 46, 0.95)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)'
              }
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <MenuItem onClick={handleProfileMenuClose}>
              <PersonOutlined sx={{ mr: 2, color: '#667eea' }} />
              <Typography variant="body2">Profile</Typography>
            </MenuItem>
            <MenuItem onClick={handleProfileMenuClose}>
              <SettingsOutlined sx={{ mr: 2, color: '#667eea' }} />
              <Typography variant="body2">Settings</Typography>
            </MenuItem>
            <MenuItem onClick={handleProfileMenuClose}>
              <Brightness4Outlined sx={{ mr: 2, color: '#667eea' }} />
              <Typography variant="body2">Dark Mode</Typography>
            </MenuItem>
            <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)', my: 1 }} />
            <MenuItem onClick={handleProfileMenuClose}>
              <ExitToAppOutlined sx={{ mr: 2, color: '#ff6b6b' }} />
              <Typography variant="body2" sx={{ color: '#ff6b6b' }}>Logout</Typography>
            </MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
  )
}

export default Topbar